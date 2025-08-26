import re
import emoji
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WhatsAppAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_selector = None
        self.feature_names = None
        
    def load_chat(self, file_path):
        """Load WhatsApp chat file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"Successfully loaded {len(lines)} lines from {file_path}")
            return lines
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            print("Please make sure the file exists in the same directory as this script.")
            return []
        except Exception as e:
            print(f"Error loading file: {e}")
            return []
    
    def parse_chat(self, lines):
        """Parse WhatsApp chat with improved regex patterns"""
        user_stats = {}
        
        # Multiple regex patterns for different WhatsApp export formats
        patterns = [
            r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2})\s*([APMapm]{2})?\s*-\s*(.*?):\s*(.*)',
            r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2})\s*-\s*(.*?):\s*(.*)',
            r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2})\]\s*(.*?):\s*(.*)',
            r'^(\d{4}-\d{2}-\d{2}),\s+(\d{1,2}:\d{2})\s*([APMapm]{2})?\s*-\s*(.*?):\s*(.*)',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if len(match.groups()) == 5:
                        user = match.group(4)
                        message = match.group(5)
                    else:
                        user = match.group(3)
                        message = match.group(4)
                    
                    # Skip system messages
                    if user in ['', 'WhatsApp'] or 'encryption' in user.lower():
                        continue
                    
                    if user not in user_stats:
                        user_stats[user] = {
                            'message_count': 0,
                            'total_length': 0,
                            'emoji_count': 0,
                            'media_count': 0,
                            'link_count': 0,
                            'question_count': 0,
                            'exclamation_count': 0,
                            'caps_count': 0,
                            'word_count': 0,
                            'unique_words': set(),
                            'night_messages': 0,
                            'day_messages': 0
                        }
                    
                    # Basic stats
                    user_stats[user]['message_count'] += 1
                    user_stats[user]['total_length'] += len(message)
                    user_stats[user]['word_count'] += len(message.split())
                    user_stats[user]['unique_words'].update(message.lower().split())
                    
                    # Emoji count (improved)
                    emoji_count = 0
                    for char in message:
                        if char in emoji.EMOJI_DATA:
                            emoji_count += 1
                    user_stats[user]['emoji_count'] += emoji_count
                    
                    # Media count
                    media_indicators = ['<media omitted>', 'media omitted', 'image omitted', 
                                      'video omitted', 'audio omitted', 'sticker omitted', 
                                      'document omitted', 'gif omitted']
                    if any(phrase in message.lower() for phrase in media_indicators):
                        user_stats[user]['media_count'] += 1
                    
                    # Link count
                    if re.search(r'http[s]?://|www\.|\.com|\.org|\.net|\.in|\.edu', message.lower()):
                        user_stats[user]['link_count'] += 1
                    
                    # Question count
                    user_stats[user]['question_count'] += message.count('?')
                    
                    # Exclamation count
                    user_stats[user]['exclamation_count'] += message.count('!')
                    
                    # Caps count
                    caps_words = sum(1 for word in message.split() if word.isupper() and len(word) > 1)
                    user_stats[user]['caps_count'] += caps_words
                    
                    # Time-based features
                    if len(match.groups()) >= 2:
                        time_str = match.group(2)
                        try:
                            hour = int(time_str.split(':')[0])
                            if hour >= 22 or hour <= 6:
                                user_stats[user]['night_messages'] += 1
                            else:
                                user_stats[user]['day_messages'] += 1
                        except:
                            user_stats[user]['day_messages'] += 1
                    
                    break
        
        print(f"Parsed messages for {len(user_stats)} users")
        return user_stats
    
    def build_features(self, user_stats):
        """Create comprehensive feature dataframe"""
        data = []
        
        for user, stats in user_stats.items():
            msg_count = stats['message_count']
            
            # Skip users with very few messages
            if msg_count < 3:
                continue
                
            avg_len = stats['total_length'] / msg_count if msg_count > 0 else 0
            avg_words = stats['word_count'] / msg_count if msg_count > 0 else 0
            vocabulary_size = len(stats['unique_words'])
            
            data.append({
                'user': user,
                'message_count': msg_count,
                'avg_message_length': avg_len,
                'avg_words_per_message': avg_words,
                'vocabulary_size': vocabulary_size,
                'emoji_count': stats['emoji_count'],
                'emoji_per_message': stats['emoji_count'] / msg_count if msg_count > 0 else 0,
                'media_count': stats['media_count'],
                'media_per_message': stats['media_count'] / msg_count if msg_count > 0 else 0,
                'link_count': stats['link_count'],
                'link_per_message': stats['link_count'] / msg_count if msg_count > 0 else 0,
                'question_count': stats['question_count'],
                'question_per_message': stats['question_count'] / msg_count if msg_count > 0 else 0,
                'exclamation_count': stats['exclamation_count'],
                'exclamation_per_message': stats['exclamation_count'] / msg_count if msg_count > 0 else 0,
                'caps_count': stats['caps_count'],
                'caps_per_message': stats['caps_count'] / msg_count if msg_count > 0 else 0,
                'night_message_ratio': stats['night_messages'] / msg_count if msg_count > 0 else 0,
                'total_words': stats['word_count'],
                'avg_word_length': avg_len / avg_words if avg_words > 0 else 0,
                'engagement_score': stats['emoji_count'] + stats['media_count'] + stats['link_count'],
                'communication_intensity': (stats['question_count'] + stats['exclamation_count']) / msg_count if msg_count > 0 else 0
            })
        
        df = pd.DataFrame(data)
        print(f"Created features for {len(df)} users")
        return df
    
    def label_users(self, df, method='percentile', threshold=0.7):
        """Label users as active/inactive with multiple methods"""
        if len(df) == 0:
            print("No data to label!")
            return df
            
        df_copy = df.copy()
        
        if method == 'percentile':
            # Use percentile-based labeling
            threshold_value = df['message_count'].quantile(threshold)
            df_copy['active'] = (df_copy['message_count'] >= threshold_value).astype(int)
        elif method == 'top_n':
            # Use top N users
            top_n = max(2, int(len(df) * 0.4))  # At least 2 users or 40%
            df_sorted = df_copy.sort_values(by='message_count', ascending=False).reset_index(drop=True)
            df_sorted['active'] = 0
            df_sorted.loc[:top_n-1, 'active'] = 1
            df_copy = df_sorted
        elif method == 'median':
            # Use median-based labeling
            median_messages = df['message_count'].median()
            df_copy['active'] = (df_copy['message_count'] >= median_messages).astype(int)
        
        active_count = df_copy['active'].sum()
        inactive_count = len(df_copy) - active_count
        print(f"Labeling: {active_count} active users, {inactive_count} inactive users")
        
        return df_copy
    
    def train_model(self, df, test_size=0.3, tune_hyperparameters=True):
        """Train Random Forest model with hyperparameter tuning"""
        if len(df) < 4:
            print("Not enough data points for training. Need at least 4 users.")
            return None, None
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in ['user', 'active']]
        X = df[feature_columns]
        y = df['active']
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Check if we have both classes
        if len(np.unique(y)) < 2:
            print("Warning: Only one class found. Adjusting labeling strategy.")
            # Redistribute labels more evenly
            threshold = 0.5
            median_messages = df['message_count'].median()
            y = (df['message_count'] >= median_messages).astype(int)
        
        # Feature selection (only if we have enough features)
        if len(feature_columns) > 8:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(8, len(feature_columns)))
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = [feature_columns[i] for i in self.feature_selector.get_support(indices=True)]
            self.feature_names = selected_features
            print(f"Selected top {len(selected_features)} features: {selected_features}")
        else:
            X_selected = X
            self.feature_names = feature_columns
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Adjust test size based on data size
        if len(df) < 10:
            test_size = 0.2
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails, do regular split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
        
        # Hyperparameter tuning
        if tune_hyperparameters and len(df) > 8:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n--- Model Performance ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        if len(X_test) > 0:
            print(f"\n--- Classification Report ---")
            print(classification_report(y_test, y_pred, zero_division=0))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n--- Feature Importance ---")
        print(feature_importance.head(10))
        
        # Cross-validation (if enough data)
        if len(df) > 6:
            try:
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=min(3, len(df)//2))
                print(f"\n--- Cross-Validation Scores ---")
                print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Cross-validation skipped: {e}")
        
        return self.model, feature_importance
    
    def analyze_chat(self, file_path, labeling_method='percentile', threshold=0.7):
        """Complete analysis pipeline"""
        print("=== WhatsApp Chat Analysis Started ===")
        
        # Load and parse chat
        lines = self.load_chat(file_path)
        if not lines:
            print("Failed to load chat file. Please check the file path and format.")
            return None
        
        user_stats = self.parse_chat(lines)
        if not user_stats:
            print("No valid messages found! Please check your chat file format.")
            return None
        
        # Build features
        df = self.build_features(user_stats)
        if len(df) < 3:
            print("Not enough users for analysis! Need at least 3 users with sufficient messages.")
            return None
        
        # Label users
        df_labeled = self.label_users(df, method=labeling_method, threshold=threshold)
        
        # Display user information
        print(f"\n--- User Analysis Summary ---")
        user_summary = df_labeled[['user', 'message_count', 'avg_message_length', 'emoji_count', 'active']].sort_values('message_count', ascending=False)
        print(user_summary.to_string(index=False))
        
        # Train model
        model, feature_importance = self.train_model(df_labeled)
        
        if model is None:
            print("Model training failed.")
            return None
        
        print("\n=== Analysis Complete ===")
        return {
            'dataframe': df_labeled,
            'model': model,
            'feature_importance': feature_importance,
            'user_stats': user_stats
        }

def main():
    """Main function to run the analysis"""
    print("WhatsApp Chat Analyzer using Random Forest")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = WhatsAppAnalyzer()
    
    # Analyze chat
    chat_file = "wp_chat.txt"  # Make sure this file exists
    
    print(f"Looking for chat file: {chat_file}")
    
    results = analyzer.analyze_chat(
        chat_file, 
        labeling_method='percentile',  # Options: 'percentile', 'top_n', 'median'
        threshold=0.7
    )
    
    if results:
        print(f"\n--- Final Results ---")
        print(f"✓ Model trained successfully!")
        print(f"✓ Dataset shape: {results['dataframe'].shape}")
        print(f"✓ Features analyzed: {len(results['feature_importance'])}")
        print(f"✓ Users classified: {len(results['dataframe'])}")
        
        # Save results
        try:
            results['dataframe'].to_csv('user_analysis_results.csv', index=False)
            print(f"✓ Results saved to 'user_analysis_results.csv'")
        except Exception as e:
            print(f"Could not save results: {e}")
            
    else:
        print("❌ Analysis failed. Please check:")
        print("   1. File 'wp_chat.txt' exists in the current directory")
        print("   2. File contains valid WhatsApp chat data")
        print("   3. Chat has at least 3 users with multiple messages")

if __name__ == "__main__":
    main()