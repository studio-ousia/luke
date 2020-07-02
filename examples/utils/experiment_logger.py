import click
import functools


def commet_logger_args(func):
    @functools.wraps(func)
    @click.option("--comet-project-name")
    @click.option("--comet-offline", is_flag=True)
    @click.option("--comet-offline-dir", type=click.Path(exists=True), default=".")
    @click.option("--comet-auto-metric-logging", is_flag=True)
    @click.option("--comet-auto-output-logging", is_flag=True)
    @click.option("--comet-log-code", is_flag=True)
    @click.option("--comet-log-env-cpu", is_flag=True)
    @click.option("--comet-log-env-gpu", is_flag=True)
    @click.option("--comet-log-env-host", is_flag=True)
    @click.option("--comet-log-graph", is_flag=True)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class NullLogger:
    def log_metric(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_parameter(self, *args, **kwargs):
        pass

    def log_parameters(self, *args, **kwargs):
        pass


class CometLogger:
    def __init__(self, args):
        import comet_ml

        comet_args = dict(
            project_name=args.comet_project_name,
            auto_metric_logging=args.comet_auto_metric_logging,
            auto_output_logging=args.comet_auto_output_logging,
            log_code=args.comet_log_code,
            log_env_cpu=args.comet_log_env_cpu,
            log_env_gpu=args.comet_log_env_gpu,
            log_env_host=args.comet_log_env_host,
            log_graph=args.comet_log_graph,
        )
        if args.comet_offline:
            self.logger = comet_ml.OfflineExperiment(offline_directory=args.comet_offline_dir, **comet_args)
        else:
            self.logger = comet_ml.Experiment(**comet_args)

    def log_metric(self, *args, **kwargs):
        self.logger.log_metric(*args, **kwargs)

    def log_metrics(self, *args, **kwargs):
        self.logger.log_metrics(*args, **kwargs)

    def log_parameter(self, *args, **kwargs):
        self.logger.log_parameter(*args, **kwargs)

    def log_parameters(self, *args, **kwargs):
        self.logger.log_parameters(*args, **kwargs)
