import re
import emoji
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# Load WhatsApp chat file
def load_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


# Parse messages and collect per-user stats
def parse_chat(lines):
    user_stats = {}

    for line in lines:
        match = re.match(r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s+\d{1,2}:\d{2}\s[APMapm]{2}\s-\s(.*?):\s(.*)', line)
        if match:
            user = match.group(2)
            message = match.group(3)

            if user not in user_stats:
                user_stats[user] = {
                    'message_count': 0,
                    'total_length': 0,
                    'emoji_count': 0,
                    'media_count': 0,
                    'link_count': 0
                }

            user_stats[user]['message_count'] += 1
            user_stats[user]['total_length'] += len(message)
            user_stats[user]['emoji_count'] += sum(1 for char in message if char in emoji.EMOJI_DATA)
            if "<Media omitted>" in message:
                user_stats[user]['media_count'] += 1
            if "http" in message or "www" in message:
                user_stats[user]['link_count'] += 1

    return user_stats


# Create feature dataframe
def build_features(user_stats):
    data = []

    for user, stats in user_stats.items():
        msg_count = stats['message_count']
        avg_len = stats['total_length'] / msg_count if msg_count > 0 else 0
        data.append({
            'user': user,
            'message_count': msg_count,
            'avg_message_length': avg_len,
            'emoji_count': stats['emoji_count'],
            'media_count': stats['media_count'],
            'link_count': stats['link_count'],
        })

    df = pd.DataFrame(data)
    return df


# Label users: top N are "active", rest are "inactive"
def label_users(df, top_n=3):
    df_sorted = df.sort_values(by='message_count', ascending=False).reset_index(drop=True)
    df_sorted['active'] = 0
    df_sorted.loc[:top_n-1, 'active'] = 1  # Top N users as active
    return df_sorted


# Train and evaluate model
def train_model(df):
    X = df[['message_count', 'avg_message_length', 'emoji_count', 'media_count', 'link_count']]
    y = df['active']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    return clf


def main():
    lines = load_chat("wp_chat.txt")
    stats = parse_chat(lines)
    df = build_features(stats)
    df_labeled = label_users(df, top_n=3)
    print("\nUser Feature Table:\n", df_labeled[['user', 'message_count', 'avg_message_length', 'emoji_count', 'active']])
    model = train_model(df_labeled)


if __name__ == "__main__":
    main()