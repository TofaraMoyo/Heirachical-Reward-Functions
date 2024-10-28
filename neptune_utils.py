import neptune
import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()


def init_neptune_run():
    return neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT_NAME"),
        api_token=os.getenv("NEPTUNE_API_TOKEN")
    )
