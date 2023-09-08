import os

import elasticapm


def get_config():
    return {
        'SERVICE_NAME': os.environ['APM_SERVICE_NAME'],
        'SERVER_URL': os.environ['APM_URI'],
        'ENVIRONMENT': os.environ['ENV'],
        'SERVICE_VERSION': os.environ['VERSION']
    }


@elasticapm.capture_span()
def set_transaction_result(result: bool):
    elasticapm.set_transaction_result(result)
