settings_1 = {
    "proportion_of_matches": 0.2,
    "link_type": "dedupe_only",
    "comparison_columns": [
        {
            "col_name": "col_1",
            "num_levels": 3,
            "m_probabilities": [0.1, 0.2, 0.7],  # Probability of typo
            "u_probabilities": [0.7, 0.1, 0.2],  # Probability of collision
        },
        {
            "col_name": "col_2",
            "m_probabilities": [0.1, 0.9],  # Probability of typo
            "u_probabilities": [0.8, 0.2],  # Probability of collision
        },
    ],
}


settings_2 = {
    "proportion_of_matches": 0.2,
    "link_type": "dedupe_only",
    "comparison_columns": [
        {
            "col_name": "col_1",
            "m_probabilities": [0.3, 0.7],  # Probability of typo
            "u_probabilities": [0.9, 0.1],  # Probability of collision
        },
        {
            "col_name": "col_2",
            "m_probabilities": [0.1, 0.9],  # Probability of typo
            "u_probabilities": [0.8, 0.2],  # Probability of collision
        },
        {
            "col_name": "col_3",
            "m_probabilities": [0.05, 0.95],  # Probability of typo
            "u_probabilities": [0.8, 0.2],  # Probability of collision
        },
    ],
}
