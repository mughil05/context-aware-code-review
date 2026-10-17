from .users import load_user

load_user(user_id)

def caller():
    return load_user(1)
