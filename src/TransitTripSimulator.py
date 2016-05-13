import numpy as np
import bisect
import copy
from collections import OrderedDict

__author__ = 'Juan Carlos Martinez Mori'


class TransitTripSimulator:

    def __init__(self, outgoing_stops, returning_stops, headway, grid_spacing, spoke_spacing, mean_grid_speed,
                 cv_grid_speed, mean_spoke_speed, cv_spoke_speed, mean_boarding_time, cv_boarding_time,
                 mean_alighting_time, cv_alighting_time, delay_control_threshold):

        # set outgoing and returning stops for the line
        self.outgoing_stops = outgoing_stops
        self.returning_stops = returning_stops

        # set all of the stops and their information in a dictionary with key
        # stop_idx: stop number in order of appearance (from beginning to end of line)
        #   'data': stop information from the input parameters
        #   'record': record of bus arrival times at the stop
        self.stops_list = self.outgoing_stops + self.returning_stops
        self.stops = OrderedDict()
        for stop_idx in range(0, len(self.stops_list)):
            self.stops[stop_idx] = {}
            if self.stops_list[stop_idx][0] != 'spoke' and self.stops_list[stop_idx][0] != 'grid':
                raise Exception('Stop type not supported.')
            self.stops[stop_idx]['data'] = self.stops_list[stop_idx]
            self.stops[stop_idx]['record'] = []

        # set headway [min]
        self.headway = headway

        # set spacings [km]
        self.grid_spacing = grid_spacing
        self.spoke_spacing = spoke_spacing

        # set mean and standard deviation of speeds [km/min]
        self.mean_grid_speed = mean_grid_speed/60
        self.std_grid_speed = cv_grid_speed*self.mean_grid_speed
        self.mean_spoke_speed = mean_spoke_speed/60
        self.std_spoke_speed = cv_spoke_speed*self.mean_spoke_speed

        # obtain mean and standard deviation of underlying normal distributions for the lognormal distributions of
        # boarding and alighting times, using the equations mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        self.mu_grid_speed = np.log((self.mean_grid_speed**2)/np.sqrt(self.std_grid_speed**2 + self.mean_grid_speed**2))
        self.sigma_grid_speed = np.sqrt(np.log(self.std_grid_speed**2/(self.mean_grid_speed**2) + 1))
        self.mu_spoke_speed = np.log((self.mean_spoke_speed**2)/np.sqrt(self.std_spoke_speed**2 + self.mean_spoke_speed**2))
        self.sigma_spoke_speed = np.sqrt(np.log(self.std_spoke_speed**2/(self.mean_spoke_speed**2) + 1))

        # set expected travel time between fixed stops [min]
        self.mean_grid_interstop_time = self.grid_spacing/self.mean_grid_speed
        self.mean_spoke_interstop_time = self.spoke_spacing/self.mean_spoke_speed

        # set mean and standard deviation of boarding and alighting times [min/pax]
        self.mean_boarding_time = mean_boarding_time/60
        self.std_boarding_time = cv_boarding_time*self.mean_boarding_time
        self.mean_alighting_time = mean_alighting_time/60
        self.std_alighting_time = cv_alighting_time*self.mean_alighting_time

        # obtain mean and standard deviation of underlying normal distributions for the lognormal distributions of
        # boarding and alighting times, using the equations mu = log((m^2)/sqrt(v+m^2)) and sigma = sqrt(log(v/(m^2)+1))
        self.mu_boarding_time = np.log((self.mean_boarding_time**2)/np.sqrt(self.std_boarding_time**2 +
                                                                            self.mean_boarding_time**2))
        self.sigma_boarding_time = np.sqrt(np.log(self.std_boarding_time**2/(self.mean_boarding_time**2) + 1))
        self.mu_alighting_time = np.log((self.mean_alighting_time**2)/np.sqrt(self.std_alighting_time**2 +
                                                                              self.mean_alighting_time**2))
        self.sigma_alighting_time = np.sqrt(np.log(self.std_alighting_time**2/(self.mean_alighting_time**2) + 1))

        # set delay control threshold [min]
        self.delay_control_threshold = delay_control_threshold

    def simulate(self, num_runs):

        # initialize lists for event list sorted by their
        # mapping to schedule list
        events = []
        schedule = []
        buses_available = 0
        added_buses = 0
        occupied_stops = np.zeros(len(self.stops_list) - 1)
        occupied_stops_matrix = np.empty(np.size(occupied_stops))
        clk_events = []

        stops = copy.deepcopy(self.stops)

        # schedule the beginning of num_runs runs
        clk_run_start = 0
        runs = OrderedDict()
        for run_idx in range(0, num_runs):
            runs[run_idx] = {}
            runs[run_idx]['travel time'] = 0
            runs[run_idx]['delay'] = 0
            runs[run_idx]['stop timetable'] = self.__get_stops_schedule(stops, clk_run_start)
            runs[run_idx]['started outgoing'] = False
            runs[run_idx]['ended outgoing'] = False
            runs[run_idx]['started returning'] = False
            runs[run_idx]['ended returning'] = False
            idx = bisect.bisect(schedule, clk_run_start)
            schedule.insert(idx, clk_run_start)
            events.insert(idx, [0, run_idx])  # 0 refers to stop no. 0, run_idx is the run number
            clk_run_start += self.headway

        # run events while there are scheduled events

        while schedule:
            occupied_stops_matrix, buses_available = self.__run_event(stops, runs, schedule, events, occupied_stops,
                                                                      occupied_stops_matrix, buses_available,
                                                                      added_buses, clk_events)

        return runs, occupied_stops_matrix, clk_events

    def __run_event(self, stops, runs, schedule, events, occupied_stops, occupied_stops_matrix, buses_available,
                    added_buses, clk_events):

        clk = schedule[0]
        event = events[0]
        clk_events.append(clk)

        # event[0] is the stop number, event[1] is the run number
        # if this is not the initial transient at stop event[0]
        if stops[event[0]]['record']:

            if event[0] == 0 and not buses_available:
                occupied_stops[event[0]] += 1
            elif event[0] == 0 and buses_available:
                buses_available -= 1
            elif event[0] == (len(self.stops_list) - 1):
                buses_available += 1
                occupied_stops[0] += 1
                occupied_stops[event[0] - 1] -= 1
            else:
                occupied_stops[event[0]] += 1
                occupied_stops[event[0] - 1] -= 1

            occupied_stops_matrix = np.vstack((occupied_stops_matrix, occupied_stops))

            # check if stop is at beginning of outgoing or returning
            if event[0] == 0:
                runs[event[1]]['started outgoing'] = True
            elif event[0] == self.outgoing_stops:
                runs[event[1]]['started returning'] = True

            # time_interarrival is the difference between current clock time and
            # clock time of last bus that arrived
            time_interarrival = clk - stops[event[0]]['record'][-1]
            stops[event[0]]['record'].append(clk)

            # delay cannot be negative
            runs[event[1]]['delay'] = max(clk - runs[event[1]]['stop timetable'][event[0]], 0)
            # if delay is beyond my control threshold, add control bus
            if runs[event[1]]['delay'] > self.delay_control_threshold and not runs[event[1]]['started returning']:
                self.__add_control_bus(event, runs, added_buses)

            # compute dwell time update travel time
            dwell_time = self.__get_dwell_time(stops, event, time_interarrival, runs, clk)
            runs[event[1]]['travel time'] += dwell_time

            # check if stop is at end of outgoing or returning
            if event[0] == len(self.outgoing_stops) - 1:
                runs[event[1]]['ended outgoing'] = True
            elif event[0] == len(self.stops_list) - 1:
                runs[event[1]]['ended returning'] = True

            if not runs[event[1]]['ended returning']:
                if stops[event[0]]['data'][0] == 'grid':
                    time_next_stop = self.grid_spacing/np.random.lognormal(self.mu_grid_speed, self.sigma_grid_speed)
                else:
                    time_next_stop = self.spoke_spacing/np.random.lognormal(self.mu_spoke_speed, self.sigma_spoke_speed)
                runs[event[1]]['travel time'] += time_next_stop
                idx = bisect.bisect(schedule, clk + time_next_stop + dwell_time)
                schedule.insert(idx, clk + time_next_stop + dwell_time)
                events.insert(idx, [event[0] + 1, event[1]])

        else:
            # set record and started/ended status for initial transient
            # set travel time and delay as nan to avoid using it in
            # output analysis
            for stop_idx in range(0, len(self.stops_list)):
                stops[stop_idx]['record'].append(runs[0]['stop timetable'][stop_idx])
            runs[event[1]]['travel time'] = float('nan')
            runs[event[1]]['delay'] = float('nan')
            runs[event[1]]['started outgoing'] = True
            runs[event[1]]['ended outgoing'] = True
            runs[event[1]]['started returning'] = True
            runs[event[1]]['ended returning'] = True

        del schedule[0]
        del events[0]

        return occupied_stops_matrix, buses_available

    def __get_stops_schedule(self, stops, clk_run_start):

        # set schedule for run
        # clk_stop is the scheduled clock time at each stop of run
        stops_schedule = []
        clk_stop = clk_run_start
        for stop in stops:
            stops_schedule.append(clk_stop)
            boarding_rate = stops[stop]['data'][1]/60
            boarding_dwell = self.headway*boarding_rate*self.mean_boarding_time
            alighting_rate = stops[stop]['data'][2]/60
            alighting_dwell = self.headway*alighting_rate*self.mean_alighting_time
            # set interstop_time accordingly
            if stops[stop]['data'][0] == 'grid':
                interstop_time = self.mean_grid_interstop_time
            else:
                interstop_time = self.mean_spoke_interstop_time
            # update clk stops
            # stop['data'][3] is the slack at each stop
            clk_stop += max(boarding_dwell, alighting_dwell) + stops[stop]['data'][3] + interstop_time

        return stops_schedule

    def __get_dwell_time(self, stops, event, clk_interarrival, runs, clk):

        # get time for bus alighting
        # event[0] is the stop number
        alighting_rate = stops[event[0]]['data'][2]/60
        alighting_time = 0
        time_for_alighting = clk_interarrival
        while True:
            time_for_alighting -= np.random.exponential(1/alighting_rate)
            if time_for_alighting > 0:
                alighting_time += np.random.lognormal(self.mu_alighting_time, self.sigma_alighting_time)
            else:
                break

        # get time for bus boarding
        # event[0] is the stop number
        boarding_rate = stops[event[0]]['data'][1]/60
        boarding_time = 0
        time_for_boarding = clk_interarrival
        while True:
            time_for_boarding -= np.random.exponential(1/boarding_rate)
            if time_for_boarding > 0:
                boarding_time += np.random.lognormal(self.mu_boarding_time, self.sigma_boarding_time)
            else:
                break

        # dwell_time is the maximum of the alighting and boarding times plus the wait for being ahead
        ahead_wait = 0
        if runs[event[1]]['stop timetable'][event[0]] - clk > 0:
            ahead_wait = runs[event[1]]['stop timetable'][event[0]] - clk
        dwell_time = max(alighting_time, boarding_time, ahead_wait)
        return dwell_time

    def __add_control_bus(self, event, runs, added_buses):

        # schedule departure of new bus at end of outgoing line
        # at end of line, 'switch'
        runs[event[1]]['started returning'] = True
        added_buses += 1

