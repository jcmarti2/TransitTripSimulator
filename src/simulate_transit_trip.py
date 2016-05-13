import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from TransitTripSimulator import TransitTripSimulator

__author__ = 'Juan Carlos Martinez Mori'

"""
this script instantiates the TransitTripSimulator class and simulates the trip taken by a bus or passenger
the number of times specified. lastly, the output is analysed and plotted
note the required parameters
:param segments: array of ordered tuples (in order of appearance in transit trip) in the format
                         [stay type, # of stays, board demand, alight demand], where stay type = {'grid stop',
                         'spoke stop', 'flexible stop', 'transfer'}, # of stays is a positive integer, board demand is
                         in [pax/min], and alight demand is in [pax/min]
:param grid_spacing: optimal spacing between grid stops [km]
:param spoke_spacing: optimal spacing between spoke stops [km]
:param flexible_area: optimal area division of flexible service [km^2]
:param fixed_headway: headway of the fixed segment of the trip [min]
:param grid_speed: mean speed for grid segments, including acceleration/deceleration but not dwell time [km/hr]
:param cv_grid_speed: coefficient of variation for grid speed []
:param spoke_speed: mean speed for spoke segments, including acceleration/deceleration but not dwell time [km/hr]
:param cv_spoke_speed: coefficient of variation for spoke speed []
:param flexible_speed: mean speed for flexible segments, including acceleration/deceleration but not dwell time [km/hr]
:param cv_flexible_speed: coefficient of variation for flexible speed []
:param boarding_time: mean boarding time per passenger [s/pax]
:param cv_boarding_time: coefficient of variation for boarding_time []
:param alighting_time: mean alighting time per passenger [s/pax]
:param cv_alighting_time: coefficient of variation for alighting_time []
:param grid_slack: time slack at each grid stop to avoid bus bunching [min]
:param spoke_slack: time slack at each spoke stop to avoid bus bunching [min]
:param fare_cost: cost of each transit ticket [$]
:param operations_cost_rate: agency costs for operation the trip per unit time [$/min]
:param delay_control_threshold: maximum allowable delay at stop arrival before a control bus is introduced [min]
:param control_bus_cost: cost of introducing a bus for delay offset [$/bus]
disclaimer: the code is provided 'as is'. it was designed with the intention of imitating reality, but it is not a
perfect representation of it (which is expected from any simulation model)
"""

# ================================================== #
#                      USER INPUT                    #
# ================================================== #
# 17 grid, 8 hs

outgoing_stops = [['spoke', 300, 1, 0], ['spoke', 300, 300, 0], ['spoke', 300, 300, 0], ['spoke', 300, 300, 0],
                  ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0],
                  ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0],
                  ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0],
                  ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0],
                  ['spoke', 300, 300, 0], ['spoke', 300, 300, 0], ['spoke', 300, 300, 0], ['spoke', 1, 300, 0]]
returning_stops = [['spoke', 300, 1, 0], ['spoke', 300, 300, 0], ['spoke', 300, 300, 0], ['spoke', 300, 300, 0],
                   ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0],
                   ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0],
                   ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0],
                   ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0], ['grid', 400, 400, 0],
                   ['spoke', 300, 300, 0], ['spoke', 300, 300, 0], ['spoke', 300, 300, 0], ['spoke', 1, 300, 0]]
grid_spacing = 0.4
spoke_spacing = 0.5
headway = 5
mean_grid_speed = 25
cv_grid_speed = 1/6
mean_spoke_speed = 25
cv_spoke_speed = 1/6
mean_boarding_time = 1
cv_boarding_time = 1/6
mean_alighting_time = 0.5
cv_alighting_time = 1/6
delay_control_threshold = 3

file_name = 'No Control'
num_runs = 100
show_animation = True
save_animation = False

# ================================================== #
#                       NO INPUT                     #
# ================================================== #

TransitTripSimulator = TransitTripSimulator(outgoing_stops, returning_stops, headway, grid_spacing, spoke_spacing,
                                            mean_grid_speed, cv_grid_speed, mean_spoke_speed, cv_spoke_speed,
                                            mean_boarding_time, cv_boarding_time, mean_alighting_time,
                                            cv_alighting_time, delay_control_threshold)
runs, occupied_stops_mat, clk_events = TransitTripSimulator.simulate(num_runs)

tts = []
for run in runs:
    tts.append(runs[run]['travel time'])
del tts[0]
mean_tt = np.mean(tts)

x_outgoing = []
y_outgoing = []
x_returning = []
y_returning = []
last_x = 0
for outgoing_stop in outgoing_stops:
    y_outgoing.append(1)
    if outgoing_stop[0] == 'grid':
        x_outgoing.append(last_x)
        last_x += grid_spacing
    elif outgoing_stop[0] == 'spoke':
        x_outgoing.append(last_x)
        last_x += spoke_spacing
    else:
        raise Exception('Stop type not supported.')
for returning_stop in returning_stops:
    y_returning.append(-1)
    if returning_stop[0] == 'grid':
        last_x -= grid_spacing
        x_returning.append(last_x)
    elif returning_stop[0] == 'spoke':
        last_x -= spoke_spacing
        x_returning.append(last_x)
    else:
        raise Exception('Stop type not supported.')
x = x_outgoing[:-1] + x_returning[:-1]
del y_outgoing[-1]
y_outgoing[0] = 0
del y_returning[-1]
y_returning[0] = 0
y = y_outgoing + y_returning

fig, ax = plt.subplots()
ax.set_title('Discrete-Event Dynamic Simulation of a Transit Trip with ' + file_name)
ax.set_xlim([-1, max(x) + 1])
ax.set_xticks([])
ax.set_yticks([])
scat = ax.scatter(x, y, color='g', label='Buses', marker='s')
scat_back = ax.scatter(x, y, s=400, marker='|', color='b', label='Bus stops')
line = ax.plot(x + [x[0]], y + [y[0]], color='k', label='Bus route')
text_xloc = sum(ax.get_xbound())/2
text_num_buses = ax.text(text_xloc, 0.45, 'Number of buses: 0', fontsize=15, horizontalalignment='center')
text_num_runs = ax.text(text_xloc, 0.30, 'Number of runs: ' + str(num_runs), fontsize=15, horizontalalignment='center')
total_dist = 2*max(x)
text_total_dist = ax.text(text_xloc, 0.15, 'Total round-trip travelled distance ' + '{0:.2f}'.format(total_dist) +
                          ' km', fontsize=15, horizontalalignment='center')
text_mean_tt = ax.text(text_xloc, 0, 'Mean worst-case travel time: ' + '{0:.2f}'.format(mean_tt) + ' min',
                       fontsize=15, horizontalalignment='center')
mean_v = 60*total_dist/mean_tt
text_mean_v = ax.text(text_xloc, -0.15, 'Mean commercial speed: ' + '{0:.2f}'.format(mean_v) + ' km/h',
                      fontsize=15, horizontalalignment='center')
texts = [text_num_buses]
ax.legend(loc='center', bbox_to_anchor=(0.5, 0.275))
num_runs = 0


def update(frame_number, mat, clk_events, texts, scat):
    occupied_stops = mat[frame_number, :].astype(int)
    s = [100*n for n in occupied_stops]
    scat.set_sizes(s)
    scat.set_color('g')
    num_buses = sum(occupied_stops)
    texts[0].set_text('Number of buses: ' + str(num_buses))
    if frame_number != 0:
        time_sleep = clk_events[frame_number] - clk_events[frame_number-1]
    else:
        time_sleep = 0
    #time.sleep(time_sleep/15)
    return scat

ani = FuncAnimation(fig, update, interval=100, frames=len(occupied_stops_mat), fargs=(occupied_stops_mat, clk_events,
                                                                                      texts, scat))

save_dir = r'C:\Users\Carlos\Desktop\418_final_project' + '\\' + file_name + '.mp4'
if show_animation:
    plt.show()
if save_animation:
    ani.save(save_dir, writer="ffmpeg", dpi=500)

