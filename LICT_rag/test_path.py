import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_env_path = os.path.abspath(os.path.join(current_dir, "..", ".env"))
print(f"-> Python is looking for your .env file EXACTLY here:\n{parent_env_path}")
print(f"-> Does the file exist at that exact path? {'YES' if os.path.exists(parent_env_path) else 'NO'}")