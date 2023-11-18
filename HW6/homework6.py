from math import log2



# # Problen 3
# # Count of each tuple (X, Y, Z)
# obs_counts = {
#     ("T", "T", "T"): 36,
#     ("T", "T", "F"): 4,
#     ("T", "F", "T"): 2,
#     ("T", "F", "F"): 8,
#     ("F", "T", "T"): 9,
#     ("F", "T", "F"): 1,
#     ("F", "F", "T"): 8,
#     ("F", "F", "F"): 32
# }
#
# # Total number of observations
# total_counts = sum(obs_counts.values())
# print("Total number of observations:", total_counts)
#
# def calculate_entropy(X, Y):
#     entropy = 0
#     for x in ("T", "F"):
#         for y in ("T", "F"):
#             # Calculate joint and marginal probabilities
#             joint_prob = sum(value for key, value in obs_counts.items() if key[X] == x and key[Y] == y) / total_counts
#             marginal_x = sum(value for key, value in obs_counts.items() if key[X] == x) / total_counts
#             marginal_y = sum(value for key, value in obs_counts.items() if key[Y] == y) / total_counts
#             # print(f"p({x}, {y}) = {joint_prob}")
#             # print(joint_prob / (marginal_x * marginal_y))
#             # Compute entropy contribution for this combination of x and y
#             entropy += joint_prob * log2(joint_prob / (marginal_x * marginal_y))
#     return entropy
#
#
# # I(X, Y), I(X, Z) and I(Z, Y)
# mutual_info_XY = calculate_entropy(0, 1)
# print("I(X, Y):", mutual_info_XY)
# mutual_info_XZ = calculate_entropy(0, 2)
# print("I(X, Z):", mutual_info_XZ)
# mutual_info_ZY = calculate_entropy(2, 1)
# print("I(Z, Y):", mutual_info_ZY)
