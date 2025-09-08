from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pyprojroot
from pydantic_settings import BaseSettings, SettingsConfigDict

class Info:
    HOME: Path = Path.home()
    BASE: Path = pyprojroot.find_root(pyprojroot.has_dir("config"))
    ENV = "dev"

load_dotenv(Path(Info.BASE, ".env"))

class Settings(BaseSettings, Info):
    model_config = SettingsConfigDict(case_sensitive=True)

    APP_NAME: str = "AutoRestore"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    ALLOWED_ORIGINS: list[str] = ["*"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    DASHSCOPE_API_KEY: str