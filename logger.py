import logging
import logging.config
import yaml

with open('configs/logger.yml', 'r') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)


def get_logger() -> logging.Logger:
    return logging.getLogger('logger')


def logger_test(logger: logging.Logger):
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')


if __name__ == '__main__':
    logger_test(get_logger())
