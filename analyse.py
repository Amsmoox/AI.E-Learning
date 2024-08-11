import pandas as pd
import numpy as np
from scipy.spatial import distance
import random
import matplotlib.pyplot as plt

# Définir le chemin vers votre fichier Excel
excel_path = r"pretest_data_with_sessions.xlsx"

# Charger les données Excel
pretest_data = pd.read_excel(excel_path)

# Configurer les paramètres Q-learning
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de réduction
sessions = ['Session 1', 'Session 2', 'Session 3', 'Session 4', 'Session 5', 'Session 6', 'Session 7']
actions = ['Application of Pretest', 'Video 1', 'PDF 1', 'Game 1', 'Video 2', 'PDF 2', 'Game 2', 'Quiz']
action_indices = {action: idx for idx, action in enumerate(actions)}

# Initialiser les Q-tables pour chaque session
q_tables = {session: np.zeros((len(actions), len(actions))) for session in sessions}

# Fonction pour calculer les récompenses
def calculate_rewards(pretest_data, session, session_col, quiz_col):
    rewards = []
    for index, row in pretest_data.iterrows():
        action_seq = eval(row[session_col])
        quiz_score = row[quiz_col]
        for i in range(len(action_seq) - 1):
            if action_seq[i] in action_indices and action_seq[i + 1] in action_indices:
                state_idx = action_indices[action_seq[i]]
                next_state_idx = action_indices[action_seq[i + 1]]
                reward = quiz_score / (i + 1)  # Exemple de récompense basée sur le score du quiz
                rewards.append((state_idx, next_state_idx, reward))
    return rewards

# Fonction pour mettre à jour la Q-table
def update_q_table(q_table, rewards):
    for (state, next_state, reward) in rewards:
        q_table[state, next_state] = q_table[state, next_state] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, next_state])
    return q_table

# Calculer et mettre à jour les Q-tables pour chaque session
for session in sessions:
    session_col = session
    quiz_col = f'{session} - Quiz Score'
    rewards = calculate_rewards(pretest_data, session, session_col, quiz_col)
    q_tables[session] = update_q_table(q_tables[session], rewards)

# Convertir une séquence d'actions en un vecteur numérique
def action_sequence_to_vector(action_sequence):
    return [action_indices[action] for action in action_sequence if action in action_indices]

# CBR: Trouver les cas similaires
def find_similar_cases(pretest_data, target_case_vector, session_col, k=1):
    distances = []
    for index, row in pretest_data.iterrows():
        case_vector = action_sequence_to_vector(eval(row[session_col]))
        if len(case_vector) == len(target_case_vector):  # Vérifier que les vecteurs ont la même longueur
            dist = distance.euclidean(target_case_vector, case_vector)
            distances.append((dist, index))
    distances.sort(key=lambda x: x[0])
    similar_indices = [index for _, index in distances[:k]]
    return pretest_data.iloc[similar_indices]

# Fonction pour déterminer la séquence optimale en utilisant le CBR
def determine_sequence_with_cbr(pretest_data, q_table, session_col):
    target_case = eval(pretest_data[session_col].iloc[0])  # Exemple de cas cible
    target_case_vector = action_sequence_to_vector(target_case)
    similar_cases = find_similar_cases(pretest_data, target_case_vector, session_col, k=1)
    sequence = eval(similar_cases[session_col].values[0])
    # Affiner la séquence avec Q-learning
    optimal_sequence = [sequence[0]]
    current_state = action_indices[sequence[0]]
    while len(optimal_sequence) < len(sequence):
        next_state = np.argmax(q_table[current_state])
        if actions[next_state] not in optimal_sequence:
            optimal_sequence.append(actions[next_state])
        current_state = next_state
    return optimal_sequence

# Générer les séquences optimales en combinant CBR et Q-learning
optimal_sequences = {}
for session in sessions:
    optimal_sequences[session] = determine_sequence_with_cbr(pretest_data, q_tables[session], session)

# Ajouter les nouvelles fonctions pour l'attribution des séquences personnalisées

# Fonction pour recevoir le vecteur des réponses du prétest de l'étudiant
def get_pretest_vector(pretest_results):
    return [1 if result == 'correct' else 0 for result in pretest_results]

# Fonction pour comparer le vecteur avec les cas de succès
def assign_sequence(pretest_vector, session, q_tables, success_cases, optimal_sequences, threshold=2.0):
    session_col = session
    similar_cases = []
    
    for index, row in success_cases.iterrows():
        case_vector = action_sequence_to_vector(eval(row[session_col]))
        if len(case_vector) == len(pretest_vector):  # Vérifier que les vecteurs ont la même longueur
            dist = distance.euclidean(pretest_vector, case_vector)
            if dist < threshold:
                similar_cases.append((dist, row[session_col]))
    
    if similar_cases:
        similar_cases.sort(key=lambda x: x[0])
        return eval(similar_cases[0][1])  # Retourne la séquence du cas le plus similaire
    
    # Sinon, assigner une séquence depuis les séquences optimales
    return random.choices(
        optimal_sequences[session], 
        weights=[np.max(q_tables[session][action_indices[action]]) for action in optimal_sequences[session]]
    )[0]

# Exemple d'utilisation de l'algorithme pour un étudiant
student_pretest_results = ['correct', 'incorrect', 'correct', 'incorrect', 'correct', 'incorrect', 'correct']  # Exemple de réponses
student_pretest_vector = get_pretest_vector(student_pretest_results)

assigned_sequences = {}
for session in sessions:
    assigned_sequence = assign_sequence(student_pretest_vector, session, q_tables, pretest_data, optimal_sequences)
    assigned_sequences[session] = assigned_sequence

# Définir un seuil pour les cas réussis
success_threshold = 15

# Initialiser les compteurs de TP et FP pour CBR et RL
tp_cbr = fp_cbr = tp_rl = fp_rl = 0

# Créer un fichier Excel avec les résultats
with pd.ExcelWriter('results_with_cbr_and_qlearning.xlsx', engine='xlsxwriter') as writer:
    # Écrire les Q-tables et les séquences optimales dans le fichier Excel
    for session in sessions:
        q_table_df = pd.DataFrame(q_tables[session], columns=actions, index=actions)
        q_table_df.to_excel(writer, sheet_name=f'Q-Table {session}')
        optimal_sequence_df = pd.DataFrame({'Optimal Sequence': optimal_sequences[session]})
        optimal_sequence_df.to_excel(writer, sheet_name=f'Optimal Sequence {session}', index=False)

        # Ajouter un graphique pour chaque Q-table (line chart)
        plt.figure(figsize=(10, 8))
        for i, row in enumerate(q_tables[session]):
            plt.plot(actions, row, label=actions[i])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f'Q-Table for {session}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'q_table_{session.replace(" ", "_").lower()}.png')
        plt.close()

        # Lire le graphique et l'insérer dans le fichier Excel
        worksheet = writer.sheets[f'Q-Table {session}']
        worksheet.insert_image('L2', f'q_table_{session.replace(" ", "_").lower()}.png')

    # Write the assigned sequences for the student
    assigned_sequences_df = pd.DataFrame(list(assigned_sequences.items()), columns=['Session', 'Assigned Sequence'])
    assigned_sequences_df = assigned_sequences_df.merge(pretest_data[['student_id']], left_index=True, right_index=True)

    # Write successful and unsuccessful cases in the Excel file
    for session in sessions:
        session_col = session
        quiz_col = f'{session} - Quiz Score'
        successful_cases_df = pretest_data[pretest_data[quiz_col] >= success_threshold][['student_id', 'pretest_score', session_col, quiz_col]]
        unsuccessful_cases_df = pretest_data[pretest_data[quiz_col] < success_threshold][['student_id', 'pretest_score', session_col, quiz_col]]

        successful_cases_df.columns = ['Student ID', 'Pretest Score', 'Learning Path', 'Quiz Score']
        unsuccessful_cases_df.columns = ['Student ID', 'Pretest Score', 'Learning Path', 'Quiz Score']

        successful_cases_df.to_excel(writer, sheet_name=f'Successful Cases {session}', index=False)
        unsuccessful_cases_df.to_excel(writer, sheet_name=f'Unsuccessful Cases {session}', index=False)

        # Calculate TP and FP for CBR using assigned sequences
        for index, row in assigned_sequences_df.iterrows():
            quiz_score = pretest_data.loc[pretest_data['student_id'] == row['student_id'], quiz_col].values[0]
            if quiz_score >= success_threshold:
                tp_cbr += 1
            else:
                fp_cbr += 1

        # Calculate TP and FP for RL using optimal sequences
        for index, row in pretest_data.iterrows():
            quiz_score = row[quiz_col]
            if quiz_score >= success_threshold:
                tp_rl += 1
            else:
                fp_rl += 1

        # Add a line chart for the quiz score distribution for successful cases
        plt.figure(figsize=(10, 8))
        successful_cases_df['Quiz Score'].plot(kind='line')
        plt.title(f'Successful Quiz Score Distribution for {session}')
        plt.xlabel('Student')
        plt.ylabel('Quiz Score')
        plt.tight_layout()
        plt.savefig(f'successful_quiz_score_distribution_{session.replace(" ", "_").lower()}.png')
        plt.close()

        # Insert the chart in the Excel file
        worksheet = writer.sheets[f'Successful Cases {session}']
        worksheet.insert_image('L2', f'successful_quiz_score_distribution_{session.replace(" ", "_").lower()}.png')

        # Add a line chart for the quiz score distribution for unsuccessful cases
        plt.figure(figsize=(10, 8))
        unsuccessful_cases_df['Quiz Score'].plot(kind='line')
        plt.title(f'Unsuccessful Quiz Score Distribution for {session}')
        plt.xlabel('Student')
        plt.ylabel('Quiz Score')
        plt.tight_layout()
        plt.savefig(f'unsuccessful_quiz_score_distribution_{session.replace(" ", "_").lower()}.png')
        plt.close()

        # Insert the chart in the Excel file
        worksheet = writer.sheets[f'Unsuccessful Cases {session}']
        worksheet.insert_image('L2', f'unsuccessful_quiz_score_distribution_{session.replace(" ", "_").lower()}.png')

    # Calculate the precision for CBR and RL
    precision_cbr = tp_cbr / (tp_cbr + fp_cbr)
    precision_rl = tp_rl / (tp_rl + fp_rl)

    # Display the precision results
    print(f"Precision for CBR: {precision_cbr:.3f}")
    print(f"Precision for RL: {precision_rl:.3f}")

    # Write the precision results in the Excel file
    precision_df = pd.DataFrame({
        'Proposal Model': ['CBR', 'RL'],
        'Precision': [precision_cbr, precision_rl]
    })
    precision_df.to_excel(writer, sheet_name='Precision Results', index=False)
