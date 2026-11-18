import dashboard_client
import rtde_control
import rtde_receive
import rtde_io as rt_io

import time
import numpy as np
import pandas as pd
import mido
from mido import MidiFile

#import robot_runner_rtde

ROBOT_IP = "192.168.1.100"#"10.165.11.242"

try: 
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    #rtde_c.setSpeedSlider(0.5) #Slows down robot for testing
    
    #Code below added so that values can be added to CSV for RL team testing
    columns = ["String Move","Force X", "Force Y", "Force Z", "Force Rx", "Force Ry", "Force Rz","Time", "Speed", "Accel"]
    df = pd.DataFrame()
    pd.DataFrame(columns=columns).to_csv("rtdeTestOne.csv", index=False)
    dataList = [0,1,2,3,4,5,6,7,8,9]
    # Code above added for RL data collection

    rtde_io = rt_io.RTDEIOInterface(ROBOT_IP)
    db = dashboard_client.DashboardClient(ROBOT_IP)
    db.connect()

    _hidden_verificationVariable = 0

    db.popup("Try initialized")
    print("Try initialized")
    rtde_c.setTcp([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # TCP = flange: all poses below are FLANGE poses
    """
    setPayload() may run into an error because of third bracket to set inertia matrix
    gemini says (mass , cog) are the only parameters of the ur_rtde function
    Can send all parameters with rtde_c.sendCustomScript("set_target_payload(0.260000, [0.050000, -0.008000, 0.024000], [0.000163, 0.000163, 0.000163, 0.000000, 0.000000, 0.000000])")
    """
    print("Attempting Payload set")
    rtde_c.setTargetPayload(0.190000, [0.128000, -0.006000, 0.019000])   # mass kg, CoG m (inertia omitted: not measured)
    #rtde_c.sendCustomScript("set_target_payload(0.260000, [0.050000, -0.008000, 0.024000], [0.000163, 0.000163, 0.000163, 0.000000, 0.000000, 0.000000])")
    print("Payload set successful")
    rtde_c.setGravity([0.0,0.0,9.82])

    #rtde_io.setToolVoltage(0)
    rtde_io.setToolDigitalOut(0,True)

    Plane_1 = [0.32571614183608727,0.7933870146316717,0.11820742021636722,-2.208244884171713,-0.24738609815952348,-1.9195085473101943]
    Plane_2 = [0.3436052139151623,0.7478032414335104,0.10077839665899768,2.768545736097797,0.1284831553507919,1.3672805628273381]
    Plane_3 = [0.31945151854627996,0.6757990361168175,0.09459088830114093,2.7301760593116526,-0.049483824131046385,1.4252239353300047]
    Plane_4 = [0.3372257363094211,0.6222230974198586,0.07001667903587716,2.621672423508307,-0.12240385890263918,1.4373011913128484]
    print("Setup Complete")
    db.closePopup()

    db.popup("Timer starting")
    time.sleep(5)
    db.closePopup()


    print("starting song")
    start_time = time.time()
    #db.popup("Started rtde song")

    event_flag = 0
    CROSS_LIFT = 0.03   # metres to lift off a string before crossing (test slowly; raise for multi-string crossings)

    # All poses: flange poses (TCP = 0), base frame, [x, y, z, rx, ry, rz] in m / rad.
    # frog_p and tip_p share one rotation (pure translation along the hair axis).
    # "out" = unit vector pointing AWAY from this string, perpendicular to the stroke
    #         (computed as stroke x string-direction; replaces the old single out_vec).
    a_bow_poses = {
        "frog_p": [0.338011, 0.705653, -0.019276, -1.367391, -2.253215, 1.585477],
        "tip_p":  [0.559952, 0.329850, 0.192236, -1.367391, -2.253215, 1.585477],
        "out":    [0.611954, 0.630295, 0.477746]
    }

    d_bow_poses = {
        "frog_p": [0.327841, 0.639624, -0.018954, -1.558290, -1.936656, 1.079513],
        "tip_p":  [0.380997, 0.154491, 0.055669, -1.558033, -1.936991, 1.080014],
        "out":    [0.751930, 0.179943, 0.634209]
    }

    g_bow_poses = {
        "frog_p": [0.309300, 0.616487, -0.038679, -1.649864, -1.700869, 0.654017],
        "tip_p":  [0.214059, 0.144070, -0.093236, -1.649864, -1.700869, 0.654017],
        "out":    [0.729636, -0.221801, 0.646866]
    }

    c_bow_poses = {
        "frog_p": [0.291029, 0.602728, -0.058844, -1.680153, -1.436804, 0.253041],
        "tip_p":  [0.103025, 0.277904, -0.203254, -1.679933, -1.437032, 0.252924],   # stroke is only 0.402 m (arm hits the stand)
        "out":    [0.589112, -0.587580, 0.554704]
    }

    pose_map = {"A": a_bow_poses, "D": d_bow_poses, "G": g_bow_poses, "C": c_bow_poses}

    def ease_scale(note_dur):
        note_dur = min(note_dur, 3.0)
        return (1.0- np.cos(np.pi * note_dur / 3.0)) / 2.0
    
    def bow(bow_dir, note_dur, bow_poses):
        #time.sleep(1)
        note_dur = min(note_dur, 3.0)
        dist_scale = ease_scale(note_dur)

        tip_p = np.array(bow_poses["tip_p"])
        frog_p = np.array(bow_poses["frog_p"])

        tcp_pose = np.array(rtde_r.getActualTCPPose())
        tcp_force = rtde_r.getActualTCPForce()
        print(f"Force: {tcp_force}")

        dataList[1:7] = tcp_force # Added for RL data collection
        dataList[7] = time.time() - start_time
        print(dataList)
        #pd.DataFrame([dataList]).to_csv("rtdeTestOne.csv", mode = "a", header = False, index = False) # Added for RL data collection

        diff_vector = tip_p[:3] - frog_p[:3]
        full_dist = np.linalg.norm(diff_vector)
        direction_vector = diff_vector / full_dist
        target_dist = dist_scale * full_dist

        if bow_dir:
            target_pose = tip_p.copy()
            #end_p = bow_poses["tip_p"]
            t_dir = 1
        else:
            target_pose = frog_p.copy()
            #end_p = bow_poses["frog_p"]
            t_dir = -1
        
        dist_to_end = np.linalg.norm(target_pose[:3]- tcp_pose[:3])

        if dist_to_end >= target_dist:
            new_xyz = tcp_pose[:3] + (t_dir * direction_vector *target_dist)
            move_p = np.concatenate([new_xyz, target_pose[3:]])
            spd,accel = get_motion_params(target_dist, note_dur)
            #print(note_dur)
            rtde_c.moveL(move_p.tolist(), speed = spd, acceleration = accel)#time= note_dur)

            dataList[8] = spd
            dataList[9] = accel
            pd.DataFrame([dataList]).to_csv("rtdeTestOne.csv", mode = "a", header = False, index = False) # A

            print(f'type: {[type(i) for i in move_p.tolist()]}')
            #rtde_c.servoL(arg0 = move_p.tolist(), arg1 = 0.0, arg2 = 0.0, arg3 = 0.1, arg4 = 0.05, arg5 = 300.0)
            #rtde_c.servoL([0.1,0.2,0.3,0.4,0.5,0.6], 0.1,0.1,0.1,0.1,300.0)
            # Range for lookahead: [0.03,0.2]       Range for gain: [100,2000]

            """
            tcp_pose = np.array(rtde_r.getActualTCPPose())
            j_pos = rtde_c.getInverseKinematics(tcp_pose)
            rtde_c.servoJ(j_pos, time = float(note_dur), lookahead_time = 0.05, gain = 300.0)
"""
        elif (dist_to_end - target_dist) >= -0.025:
            spd, accel = get_motion_params(dist_to_end, note_dur)
            #print(note_dur)
            rtde_c.moveL(target_pose.tolist(), speed = spd, acceleration = accel)#time = note_dur)


            dataList[8] = spd
            dataList[9] = accel
            pd.DataFrame([dataList]).to_csv("rtdeTestOne.csv", mode = "a", header = False, index = False) # A


            print(f'type: {[type(i) for i in target_pose.tolist()]}')
            #rtde_c.servoL(arg0 = target_pose.tolist(), arg1 = 0.0, arg2 = 0.0, arg3 = 0.1, arg4 = 0.05, arg5 = 300.0)
            #rtde_c.servoL([0.1,0.2,0.3,0.4,0.5,0.6], 0.1,0.1,0.1,0.1,300.0)
            # Range for lookahead: [0.03,0.2]       Range for gain: [100,2000]
            """
            tcp_pose = np.array(rtde_r.getActualTCPPose())
            j_pos = rtde_c.getInverseKinematics(tcp_pose)
            rtde_c.servoJ(j_pos, time = float(note_dur), lookahead_time = 0.05, gain = 300.0)
        """
        else:
            d1 = dist_to_end
            d2 = target_dist - d1
            #print(note_dur)
            time_a = note_dur * (d1 / target_dist)
            time_b = note_dur * (d2 / target_dist)
            spd1, accel1 = get_motion_params(d1, time_a)
            print(f'type: {[type(i) for i in target_pose.tolist()]}')
            rtde_c.moveL(target_pose.tolist(), speed = spd1, acceleration = accel1)#time = time_a)


            dataList[8] = spd1
            dataList[9] = accel1
            pd.DataFrame([dataList]).to_csv("rtdeTestOne.csv", mode = "a", header = False, index = False) # A

            #rtde_c.servoL(arg0 = target_pose.tolist(), arg1 = 0.0, arg2 = 0.0, arg3 = 0.1, arg4 = 0.05, arg5 = 300.0)
            #rtde_c.servoL([0.1,0.2,0.3,0.4,0.5,0.6], 0.1,0.1,0.1,0.1,300.0)
            # Range for lookahead: [0.03,0.2]       Range for gain: [100,2000]
            """
            tcp_pose = np.array(rtde_r.getActualTCPPose())
            j_pos = rtde_c.getInverseKinematics(tcp_pose)
            rtde_c.servoJ(j_pos, time = float(note_dur), lookahead_time = 0.05, gain = 300.0)
            """
            new_xyz = target_pose[:3] + (-1 * t_dir * direction_vector * d2)
            move_p_final = np.concatenate([new_xyz, target_pose[3:]])
            
            spd2, accel2 = get_motion_params(d2, time_b)
            rtde_c.moveL(move_p_final.tolist(), speed = spd2, acceleration = accel2)#time = time_b)


            dataList[8] = spd2
            dataList[9] = accel2
            pd.DataFrame([dataList]).to_csv("rtdeTestOne.csv", mode = "a", header = False, index = False) # A

            #rtde_c.servoL(arg0 = move_p_final.tolist(), arg1 = 0.0, arg2 = 0.0, arg3 = 0.1, arg4 = 0.05, arg5 = 300.0)
            #rtde_c.servoL([0.1,0.2,0.3,0.4,0.5,0.6], 0.1,0.1,0.1,0.1,300.0)
            # Range for lookahead: [0.03,0.2]       Range for gain: [100,2000]
            """
            tcp_pose = np.array(rtde_r.getActualTCPPose())
            j_pos = rtde_c.getInverseKinematics(tcp_pose)
            rtde_c.servoJ(j_pos, time = float(note_dur), lookahead_time = 0.05, gain = 300.0)
            """
    def get_motion_params(d , t):
            if d < 0.0001 or t < 0.0001 : return 0.1,0.5
            v = 1.5 * (d/t) #(2.0*d)/t
            a = 4.5 * (d / (t**2)) # (2.0*d)/(t**2)
            return min(v,1.5), min(a,5.0)

    def stay():
        rtde_c.stopJ(2.5)
        time.sleep(0.025)

    def a_bow(bow_dir, note_dur):
        event_flag = 1
        #rtde_c.setStandardAnalogOut(0, event_flag)
        #rtde_c.setStandardAnalogOut(1, int(bow_dir))
        bow(bow_dir, note_dur, a_bow_poses)

        event_flag = 2
        #rtde_c.setStandardAnalogOut(0, event_flag)

    def d_bow(bow_dir, note_dur):
        event_flag = 3
        #rtde_c.setStandardAnalogOut(1, int(bow_dir))
        #rtde_c.setStandardAnalogOut(0, event_flag)

        bow(bow_dir, note_dur, d_bow_poses)

        event_flag = 4
        #rtde_c.setStandardAnalogOut(0, event_flag)

    def g_bow(bow_dir, note_dur):
        event_flag = 5
        #rtde_c.setStandardAnalogOut(1, int(bow_dir))
        #rtde_c.setStandardAnalogOut(0, event_flag)

        bow(bow_dir, note_dur, g_bow_poses)

        event_flag = 6
        #rtde_c.setStandardAnalogOut(0 , event_flag)

    def c_bow(bow_dir, note_dur):
        event_flag = 7
        #rtde_c.setStandardAnalogOut(1, int(bow_dir))
        #rtde_c.setStandardAnalogOut(0, event_flag)

        bow(bow_dir, note_dur, c_bow_poses)

        event_flag = 8
        #rtde_c.setStandardAnalogOut(0 , event_flag)

    def string_crossing(start_bow_poses, end_bow_poses, next_dir):
        tcp_pose = np.array(rtde_r.getActualTCPPose())

        # Lift straight off the string we are leaving, along THAT string's own "out" direction.
        out_vec = np.array(start_bow_poses["out"], dtype=float)
        step1 = tcp_pose.copy()
        step1[:3] = tcp_pose[:3] + out_vec * CROSS_LIFT

        target_1 = np.concatenate([step1[:3], tcp_pose[3:]])
        rtde_c.moveL(target_1.tolist())
        #rtde_c.servoL(target_1.tolist())#, lookahead_time = 0.05, gain = 300.0)
        # Range for lookahead: [0.03,0.2]       Range for gain: [100,2000]

        # Rotate (while clear of the strings) to the rotation of the string we are going TO.
        end_rot = end_bow_poses["frog_p"][3:]
        target_2 = np.concatenate([step1[:3], end_rot])
        rtde_c.moveL(target_2.tolist())
        #rtde_c.servoL(target_2.tolist())# lookahead_time = 0.05, gain = 300.0)
        # Range for lookahead: [0.03,0.2]       Range for gain: [100,2000]

        current_p = np.array(rtde_r.getActualTCPPose())

        tip = np.array(end_bow_poses["tip_p"])
        frog = np.array(end_bow_poses["frog_p"])

        bow_vec = tip[:3] - frog[:3]
        direction_vector = bow_vec / np.linalg.norm(bow_vec)

        A = tip[:3]
        start_xyz = current_p[:3]

        relative_pos = start_xyz - A
        stroke_len = np.linalg.norm(bow_vec)
        projection_dist = np.dot(relative_pos, direction_vector)
        projection_dist = float(np.clip(projection_dist, -stroke_len, 0.0))   # stay inside the taught stroke
        target_xyz = (projection_dist * direction_vector) + A

        final_target = np.concatenate([target_xyz, tip[3:]])

        #NEW CODE FOR SPEED AND ACCEL CALCULATIONS BELOW
        # No duration value in this function, need to find time to calculate this movement
        #distance = np.linalg.norm(final_target[:3]-current_p[:3])
        #safe_dur = max(, 0.01)

        rtde_c.moveL(final_target.tolist(), speed = 0.25, acceleration = 1.2) #speed changed from 3.67 to 0.25
        
        #rtde_c.servoL(final_target.tolist())#, lookahead_time = 0.05, gain = 300.0)
        # Range for lookahead: [0.03,0.2]       Range for gain: [100,2000]

    def a_to_d(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 101)
        string_crossing(a_bow_poses, d_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0, 102)
    
    def a_to_g(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 109)
        string_crossing(a_bow_poses, g_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,110)

    def d_to_a(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 103)
        string_crossing(d_bow_poses, a_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,104)

    def d_to_g(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 105)
        string_crossing(d_bow_poses, g_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,106)

    def d_to_c(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 113)
        string_crossing(d_bow_poses, c_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,114)

    def g_to_a(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 111)
        string_crossing(g_bow_poses, a_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,112)

    def g_to_d(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 107)
        string_crossing(g_bow_poses, d_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,108)

    def g_to_c(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 117)
        string_crossing(g_bow_poses, c_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,118)

    def c_to_d(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 115)
        string_crossing(c_bow_poses, d_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,116)

    def c_to_g(next_dir = 0):
        #rtde_c.setStandardAnalogOut(0, 119)
        string_crossing(c_bow_poses, g_bow_poses, next_dir)
        #rtde_c.setStandardAnalogOut(0,120)

    #THE CODE ABOVE THIS WAS "song.script" THAT WAS CONVERTED TO ur_rtde

    #Code below will be used to replicate robot_runner.py
    def run_performance(midi_path):
        time = 0
        x = 0
        #notes = [0]
        notes = [{'note': 'A3', 'duration': 2.0, 'string': 'A'}, {'note': 'cross', 'duration': 0.0, 'string': 'A-D'}, {'note': 'D3', 'duration': 2.0, 'string': 'D'}, {'note': 'cross', 'duration': 0.0, 'string': 'D-G'}, {'note': 'G3', 'duration': 2.0, 'string': 'G'}]
        # while x < 1:
        #     notes[x] = {'number': 50, 'note': 'D3', 'duration': 2.0, 'string': 'G', 'start_time': time, 'end_time': time + 479}
        #     x += 1
        #     time += 480

        #print(notes)
        #notes = [{'number': 50, 'note': 'D3', 'duration': 0.9979166666666667, 'string': 'D', 'start_time': 0, 'end_time': 479}]
        
        #notes = [{'number': 50, 'note': 'A3', 'duration': 0.9979166666666667, 'string': 'A', 'start_time': 0, 'end_time': 479}]
        #,{'number': 50, 'note': 'D3', 'duration': 0.9979166666666667, 'string': 'D', 'start_time': 480, 'end_time': 959}
        #robot_runner_rtde.parse_midi(midi_path)
        #print(notes)
        
        first_string = notes[0]["string"]
        rtde_c.moveJ(rtde_c.getInverseKinematics(pose_map[first_string]["frog_p"]))

        bow_dir = True
        for note in notes:
            string_key = note["string"]
            if "-" in string_key:
                s1, s2 = string_key.split("-")
                print(f"Crossing from {s1} to {s2}")
                dataList[0] = f"Cross {s1} to {s2}" # Added for RL data collection
                string_crossing(pose_map[s1],pose_map[s2],0)
            
            else:
                bow_dir = not bow_dir
                print(f"Playing {note['note']} on {string_key}")
                dataList[0] = f"{note['note']} on {string_key}" # Added for RL data collection
                #rtde_c.setStandardAnalogOut(0, 1)
                bow(bow_dir, note["duration"], pose_map[string_key])
                rtde_c.stopJ(2.5)
                #rtde_c.setStandardAnalogOut(0, 2)
        
        
        rtde_c.disconnect()

    db.popup("Running RTDE Script")
    run_performance("../MIDI-Files/twinkle_twinkle-open.mid")
except Exception as e:
    print(f"An error has occured: {e}")
finally:
    print("Finally initialized")
    rtde_c.stopScript()
    rtde_c.disconnect()
