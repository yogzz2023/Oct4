import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    # ... [Previous CVFilter implementation remains the same]

def read_measurements_from_csv(file_path):
    # ... [Previous implementation remains the same]

def sph2cart(az, el, r):
    # ... [Previous implementation remains the same]

def cart2sph(x, y, z):
    # ... [Previous implementation remains the same]

def form_measurement_groups(measurements, max_time_diff=0.050):
    # ... [Previous implementation remains the same]

def form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold):
    # ... [Previous implementation remains the same]

def mahalanobis_distance(track, report, cov_inv):
    # ... [Previous implementation remains the same]

def select_best_report(cluster_tracks, cluster_reports, kalman_filter):
    # ... [Previous implementation remains the same]

def select_initiation_mode(mode):
    # ... [Previous implementation remains the same]

def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    # ... [Previous implementation remains the same]

def correlation_check(track, measurement, doppler_threshold, range_threshold):
    last_measurement = track['measurements'][-1][0]
    last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
    measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
    distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
    
    doppler_correlated = doppler_correlation(measurement[4], last_measurement[4], doppler_threshold)
    range_satisfied = distance < range_threshold
    
    return doppler_correlated and range_satisfied

def initialize_filter_state(kalman_filter, x, y, z, vx, vy, vz, time):
    kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)

def perform_jpda(tracks, reports, kalman_filter):
    clusters = form_clusters_via_association(tracks, reports, kalman_filter, kalman_filter.gate_threshold)
    best_reports = []
    for cluster_tracks, cluster_reports in clusters:
        best_track_idx, best_report = select_best_report(cluster_tracks, cluster_reports, kalman_filter)
        if best_report is not None:
            best_reports.append((best_track_idx, best_report))
    return best_reports

def main():
    file_path = 'ttk.csv'
    measurements = read_measurements_from_csv(file_path)

    kalman_filter = CVFilter()
    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    tracks = []
    track_id_list = []
    filter_states = []

    doppler_threshold = 100
    range_threshold = 100
    firm_threshold = 3
    mode = '3-state'

    firm_threshold = select_initiation_mode(mode)
    # Initialize variables outside the loop
    miss_counts = {}
    hit_counts = {}
    firm_ids = set()
    state_map = {}
    progression_states = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
    }[firm_threshold]

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")

        if len(group) == 1:  # Single measurement
            measurement = group[0]
            assigned = False
            for track_id, track in enumerate(tracks):
                if correlation_check(track, measurement, doppler_threshold, range_threshold):
                    current_state = state_map.get(track_id, None)
                    if current_state == 'Poss1':
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])
                    elif current_state == 'Tentative1':
                        last_measurement = track['measurements'][-1][0]
                        dt = measurement[3] - last_measurement[3]
                        vx = (sph2cart(*measurement[:3])[0] - sph2cart(*last_measurement[:3])[0]) / dt
                        vy = (sph2cart(*measurement[:3])[1] - sph2cart(*last_measurement[:3])[1]) / dt
                        vz = (sph2cart(*measurement[:3])[2] - sph2cart(*last_measurement[:3])[2]) / dt
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), vx, vy, vz, measurement[3])
                    elif current_state == 'Firm':
                        kalman_filter.predict_step(measurement[3])
                        kalman_filter.update_step(np.array(sph2cart(*measurement[:3])).reshape(3, 1))
                    
                    track['measurements'].append((measurement, current_state))
                    assigned = True
                    break

            if not assigned:
                new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                if new_track_id is None:
                    new_track_id = len(track_id_list)
                    track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                else:
                    track_id_list[new_track_id]['state'] = 'occupied'
                
                tracks.append({
                    'track_id': new_track_id,
                    'measurements': [(measurement, 'Poss1')],
                    'current_state': 'Poss1'
                })
                state_map[new_track_id] = 'Poss1'
                initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])

        else:  # Multiple measurements
            reports = [sph2cart(*m[:3]) for m in group]
            best_reports = perform_jpda([track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter)

            for track_id, best_report in best_reports:
                current_state = state_map.get(track_id, None)
                if current_state == 'Poss1':
                    initialize_filter_state(kalman_filter, *best_report, 0, 0, 0, group[0][3])
                elif current_state == 'Tentative1':
                    last_measurement = tracks[track_id]['measurements'][-1][0]
                    dt = group[0][3] - last_measurement[3]
                    vx = (best_report[0] - sph2cart(*last_measurement[:3])[0]) / dt
                    vy = (best_report[1] - sph2cart(*last_measurement[:3])[1]) / dt
                    vz = (best_report[2] - sph2cart(*last_measurement[:3])[2]) / dt
                    initialize_filter_state(kalman_filter, *best_report, vx, vy, vz, group[0][3])
                elif current_state == 'Firm':
                    kalman_filter.predict_step(group[0][3])
                    kalman_filter.update_step(np.array(best_report).reshape(3, 1))
                
                tracks[track_id]['measurements'].append((cart2sph(*best_report) + (group[0][3], group[0][4]), current_state))

            # Handle unassigned measurements
            assigned_reports = set(best_report for _, best_report in best_reports)
            for report in reports:
                if tuple(report) not in assigned_reports:
                    new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                    if new_track_id is None:
                        new_track_id = len(track_id_list)
                        track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                    else:
                        track_id_list[new_track_id]['state'] = 'occupied'
                    
                    tracks.append({
                        'track_id': new_track_id,
                        'measurements': [(cart2sph(*report) + (group[0][3], group[0][4]), 'Poss1')],
                        'current_state': 'Poss1'
                    })
                    state_map[new_track_id] = 'Poss1'
                    initialize_filter_state(kalman_filter, *report, 0, 0, 0, group[0][3])

        # Update states based on hit counts
        for track_id, track in enumerate(tracks):
            hit_counts[track_id] = hit_counts.get(track_id, 0) + 1
            if hit_counts[track_id] >= firm_threshold:
                state_map[track_id] = 'Firm'
                firm_ids.add(track_id)
            elif hit_counts[track_id] == 2:
                state_map[track_id] = 'Tentative1'
            track['current_state'] = state_map[track_id]

    # Print summary
    for track_id, track in enumerate(tracks):
        print(f"Track {track_id}:")
        print(f"  Current State: {track['current_state']}")
        print(f"  Number of Measurements: {len(track['measurements'])}")
        print(f"  Hit Count: {hit_counts.get(track_id, 0)}")
        print(f"  Miss Count: {miss_counts.get(track_id, 0)}")
        print(f"  Last Measurement: {track['measurements'][-1][0]}")
        print()

if __name__ == "__main__":
    main()