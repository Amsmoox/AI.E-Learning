import pandas as pd
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Define the structure for 30 students
data_structure = {
    'student_id': [f'Student_{i+1}' for i in range(30)],
    'pretest_score': np.random.normal(loc=6, scale=2, size=30)  # Normal distribution centered at 6 with a standard deviation of 2
}

# Create a DataFrame
pretest_data = pd.DataFrame(data_structure)

# Ensure scores are within a realistic range (e.g., 0-20)
pretest_data['pretest_score'] = pretest_data['pretest_score'].clip(0, 20)

# Define learning sequences for Session 1 according to the article
learning_sequences_s1 = {
    '0-5': [
        ['a', 'b1', 'c1', 'd1', 'e']  # 3 basic resources
    ],
    '6-7': [
        ['a', 'b1', 'c1', 'd2', 'e'],  # 2 basic, 1 advanced
        ['a', 'b1', 'c2', 'd1', 'e'],
        ['a', 'b2', 'c1', 'd1', 'e']
    ],
    '8-9': [
        ['a', 'b1', 'c2', 'd2', 'e'],  # 1 basic, 2 advanced
        ['a', 'b2', 'c2', 'd1', 'e'],
        ['a', 'b2', 'c1', 'd2', 'e']
    ],
    '10-20': [
        ['a', 'b2', 'c2', 'd2', 'e']  # 3 advanced resources
    ]
}

# Define fixed resources for Sessions 2 to 7
session_resources = {
    2: ['a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'd'],
    3: ['a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'd'],
    4: ['a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'd'],
    5: ['a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'd'],
    6: ['a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'd'],
    7: ['a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'd']
}

# Define the resource to section mapping for clarity
resource_mapping_s1 = {
    'a': 'Application of Pretest',
    'b1': 'Video 1',
    'b2': 'Video 2',
    'c1': 'Game 1',
    'c2': 'Game 2',
    'd1': 'PDF 1',
    'd2': 'PDF 2',
    'e': 'Quiz'
}

resource_mapping_s2_7 = {
    'a1': 'Video 1',
    'a2': 'Video 2',
    'b1': 'Game 1',
    'b2': 'Game 2',
    'c1': 'PDF 1',
    'c2': 'PDF 2',
    'd': 'Quiz'
}

# Function to assign learning sequence for Session 1 based on pretest score
def assign_sequence_s1(pretest_score):
    if 0 <= pretest_score < 6:
        sequences = learning_sequences_s1['0-5']
    elif 6 <= pretest_score < 8:
        sequences = learning_sequences_s1['6-7']
    elif 8 <= pretest_score < 10:
        sequences = learning_sequences_s1['8-9']
    elif pretest_score >= 10:
        sequences = learning_sequences_s1['10-20']
    else:
        sequences = learning_sequences_s1['0-5']  # Default to 0-5 if outside range

    return sequences[np.random.choice(len(sequences))]  # Randomly choose a sequence from the list

# Assign learning sequences for Session 1
pretest_data['Session 1'] = pretest_data['pretest_score'].apply(assign_sequence_s1)

# Convert learning sequence to detailed path with resource names for Session 1
pretest_data['Session 1'] = pretest_data['Session 1'].apply(lambda seq: [resource_mapping_s1[res] for res in seq])

# Assign fixed resources for Sessions 2 to 7 and ensure 'Quiz' is at the end
for session in range(2, 8):
    resources = session_resources[session][:-1]  # Remove 'Quiz' for intermediate steps
    np.random.shuffle(resources)  # Randomize the order
    resources.append('d')  # Add 'Quiz' at the end
    pretest_data[f'Session {session}'] = [resources for _ in range(pretest_data.shape[0])]
    pretest_data[f'Session {session}'] = pretest_data[f'Session {session}'].apply(lambda seq: [resource_mapping_s2_7[res] for res in seq])

# Assign a single quiz score (0-20) for each session
for session in range(1, 8):
    pretest_data[f'Session {session} - Quiz Score'] = np.random.randint(0, 21, pretest_data.shape[0])

# Print the DataFrame
print(pretest_data)

# Export data to JSON
json_path = 'pretest_data_with_sessions.json'
pretest_data.to_json(json_path, orient='records', indent=4)
print(f'Data exported to JSON: {json_path}')

# Export data to Excel
excel_path = 'pretest_data_with_sessions.xlsx'
pretest_data.to_excel(excel_path, index=False)
print(f'Data exported to Excel: {excel_path}')
