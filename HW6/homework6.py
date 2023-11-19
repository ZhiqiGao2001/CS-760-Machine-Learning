from math import log2
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination


alarm_model = BayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_burglary = TabularCPD(
    variable="Burglary", variable_card=2, values=[[0.9], [0.1]]
)
cpd_earthquake = TabularCPD(
    variable="Earthquake", variable_card=2, values=[[0.8], [0.2]]
)
cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.9, 0.7, 0.2, 0.1], [0.1, 0.3, 0.8, 0.9]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
)
cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.8, 0.1], [0.2, 0.9]],
    evidence=["Alarm"],
    evidence_card=[2],
)
cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.9, 0.3], [0.1, 0.7]],
    evidence=["Alarm"],
    evidence_card=[2],
)

# Associating the parameters with the model structure
alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls
)

print(alarm_model.check_model())

infer = VariableElimination(alarm_model)

print(alarm_model.get_cpds("Burglary"))
print(alarm_model.get_cpds("Earthquake"))
print(alarm_model.get_cpds("Alarm"))
print(alarm_model.get_cpds("JohnCalls"))
print(alarm_model.get_cpds("MaryCalls"))

print("P(Burglary|JohnCalls=1, MaryCalls=1, Earthquake=0)")
print(infer.query(variables=["Burglary"], evidence={"Earthquake": 0, "JohnCalls": 1, "MaryCalls": 1}))

print("P(Burglary|JohnCalls=1, MaryCalls=1, Earthquake=1)")
print(infer.query(variables=["Burglary"], evidence={"Earthquake": 1, "JohnCalls": 1, "MaryCalls": 1}))


# Problen 3
# Count of each tuple (X, Y, Z)
obs_counts = {
    ("T", "T", "T"): 36,
    ("T", "T", "F"): 4,
    ("T", "F", "T"): 2,
    ("T", "F", "F"): 8,
    ("F", "T", "T"): 9,
    ("F", "T", "F"): 1,
    ("F", "F", "T"): 8,
    ("F", "F", "F"): 32
}

# Total number of observations
total_counts = sum(obs_counts.values())
print("Total number of observations:", total_counts)

def calculate_entropy(X, Y):
    entropy = 0
    for x in ("T", "F"):
        for y in ("T", "F"):
            # Calculate joint and marginal probabilities
            joint_prob = sum(value for key, value in obs_counts.items() if key[X] == x and key[Y] == y) / total_counts
            marginal_x = sum(value for key, value in obs_counts.items() if key[X] == x) / total_counts
            marginal_y = sum(value for key, value in obs_counts.items() if key[Y] == y) / total_counts
            # print(f"p({x}, {y}) = {joint_prob}")
            # print(joint_prob / (marginal_x * marginal_y))
            # Compute entropy contribution for this combination of x and y
            entropy += joint_prob * log2(joint_prob / (marginal_x * marginal_y))
    return entropy


# I(X, Y), I(X, Z) and I(Z, Y)
mutual_info_XY = calculate_entropy(0, 1)
print("I(X, Y):", mutual_info_XY)
mutual_info_XZ = calculate_entropy(0, 2)
print("I(X, Z):", mutual_info_XZ)
mutual_info_ZY = calculate_entropy(2, 1)
print("I(Z, Y):", mutual_info_ZY)
