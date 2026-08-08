import logging

import hydra
from omegaconf import DictConfig

try:
    from navsim_origin.planning.script.bootstrap_navsim_origin import force_navsim_origin_alias
except ImportError:
    from bootstrap_navsim_origin import force_navsim_origin_alias

force_navsim_origin_alias()

from navsim.planning.script.external_bev_export_utils import run_external_bev_export


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Starting external BEV PNG export (processed_scenes + nuScenes)")
    exported = run_external_bev_export(cfg)
    logger.info("External BEV PNG export completed with %d samples", exported)


if __name__ == "__main__":
    main()
