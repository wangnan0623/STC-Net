from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.eotb_path = ''
    settings.fe108_path = '/home/Data/FE240/test'
    settings.visevent_path = '/home/Data/VisEvent/test_subset'
    settings.test_subset_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/wangnan/project/AFNet-main/logs/test/checkpoints/ltr/ahnet/ahnet'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/home/wangnan/project/AFNet-main/logs/VisEvent_baseline/plot'
    settings.results_path = '/home/wangnan/project/AFNet-main/logs/VisEvent_baseline/results'    # Where to store tracking results
    settings.segmentation_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

