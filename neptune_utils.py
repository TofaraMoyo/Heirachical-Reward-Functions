import neptune
import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()


def init_neptune_run():
    return neptune.init_run(
        project="DewaSai/moyo",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMGQ2MzkzMi04OTEyLTQ5NDItOWRkZi0zZDE4MDVlZDA4OTIifQ=="
    )
