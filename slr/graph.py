import torch


# POSE
# 0 as root (Constant)
# The number gets bigger when it goes farther from 0.
POSE_MAP = [
    (0, 15),
    (0, 16),
    (15, 17),
    (16, 18),
    (0, 1),
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 22),
    (22, 23),
    (11, 24),
    (8, 12),
    (12, 13),
    (13, 14),
    (14, 19),
    (19, 20),
    (14, 21)
]

POSE_ADJ_MATRIX = torch.zeros((25, 25))
for (i, j) in POSE_MAP:
    POSE_ADJ_MATRIX[i, j] = 1

POSE_POWERED_MATRICES = [POSE_ADJ_MATRIX]

for i in range(6):
    POSE_POWERED_MATRICES.append(torch.matmul(POSE_POWERED_MATRICES[-1], POSE_ADJ_MATRIX))

POSE_CHILD_MATRIX = torch.sum(torch.stack(POSE_POWERED_MATRICES, dim=0), dim=0) + torch.eye(25)

# Same goes for both hands

HAND_MAP = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

HAND_ADJ_MATRIX = torch.zeros((21, 21))
for (i, j) in HAND_MAP:
    HAND_ADJ_MATRIX[i, j] = 1

HAND_POWERED_MATRICES = [HAND_ADJ_MATRIX]

for i in range(4):
    HAND_POWERED_MATRICES.append(torch.matmul(HAND_POWERED_MATRICES[-1], HAND_ADJ_MATRIX))

HAND_CHILD_MATRIX = torch.sum(torch.stack(HAND_POWERED_MATRICES, dim=0), dim=0) + torch.eye(21)


FACE_MAP = torch.arange(40)
FACE_MAP = FACE_MAP.reshape([-1, 1])
FACE_MAP = torch.cat([FACE_MAP, FACE_MAP], dim=1)