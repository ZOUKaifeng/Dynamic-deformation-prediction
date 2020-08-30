import numpy as np
from transforms3d.quaternions import axangle2quat,qmult, qnorm, qconjugate




def axis_ang2quater(axis_ang):
    a = np.linalg.norm(axis_ang, axis=1)
    a = np.expand_dims(a, axis = 1)
    u = np.divide(axis_ang,a)
    flag = np.isnan(u)
    u[flag == 1] = 0
    quater = np.concatenate((np.cos(a/2), u*np.sin(a/2)), axis=1)
    #quter_1 = axangle2quat(u, a)
    return quater


def quater2ang_v(quater_1, quater_2, dt):
    delta_q =qmult(quater_2, qconjugate(quater_1))
    delta_q_len = np.linalg.norm(delta_q[1:])
    delta_q_angle = 2*np.arctan2(delta_q_len, delta_q[0])
    w = delta_q[1:] * delta_q_angle * 1/dt

    return w


def compute_ang_v(poses, t):

    angular_velocity = []
    #sequence = poses
    #sequence = data['poses'][:, 3:66]
    for sequence in poses:
        sequence = np.asarray(sequence).astype('float32')
        seq_velocity = np.zeros((sequence.shape[0]-1, 22 * 3))
        for frameID, frame in enumerate(sequence[:-1]):
            frame = frame.reshape((22, 3))  # pose of frameID
            next_frame = sequence[frameID+1].reshape((22, 3)) #pose of frameID+1

            current_quater = axis_ang2quater(frame)
            next_quater = axis_ang2quater(next_frame)

            v = np.zeros(22 * 3)
            for f, quater in enumerate(current_quater):
                next_qua = next_quater[f]
                w = quater2ang_v(quater, next_qua, dt = t)
                v[3*f:3*f+3] = w
               
            seq_velocity[frameID,:] = v
        angular_velocity.append(seq_velocity)
    return angular_velocity



def compute_acceleration(angular_velocity, t):
    angular_acc = []
    for sequence in angular_velocity:
        sequence = np.asarray(sequence)
        acc_sequence = np.zeros((sequence.shape[0]-1,66))
        for frameID, frame in enumerate(sequence[:-1]):
            next_frame = sequence[frameID+1]
            d_frame = next_frame - frame
            dt = t
            acc = d_frame/dt
            acc_sequence[frameID,:] = acc
        angular_acc.append(acc_sequence)
    return angular_acc

def compute_root_va(trans, t):
    t = 1/t
    
    m = []
    for tran in trans:
        tran = np.asarray(tran).astype('float32')
        l = tran.shape[0]-1
        d = (tran[1:] - tran[0:l]).squeeze()
        s = np.zeros((l,6))
        s[:,0:3] = d/t
        s[:,3:6] = 2*d/(t**2)


        m.append(s)
    return m