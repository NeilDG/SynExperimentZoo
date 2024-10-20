import global_config

class ConfigHolder():
    _sharedInstance = None

    @staticmethod
    def initialize(yaml_data, hyperparam_data, weight_data):
        if(ConfigHolder._sharedInstance == None):
            ConfigHolder._sharedInstance = ConfigHolder(yaml_data, hyperparam_data, weight_data)

    @staticmethod
    def destroy():
        ConfigHolder._sharedInstance = None

    @staticmethod
    def getInstance():
        return ConfigHolder._sharedInstance

    def __init__(self, yaml_data, hyper_param_data, weight_data):
        global_config.loaded_network_config = yaml_data
        self.hyper_param_config = hyper_param_data
        self.loss_weights_config = weight_data

    def get_network_config(self):
        return global_config.loaded_network_config

    def get_network_attribute(self, key, default):
        if(key in global_config.loaded_network_config):
            # print("Key ", key, " found. Returning ", global_config.loaded_network_config[key])
            return global_config.loaded_network_config[key]
        else:
            return default

    def get_loss_weights(self):
        return self.loss_weights_config

    def get_loss_weight_by_key(self, key):
        loss_weights_table = self.loss_weights_config["loss_weights"][global_config.loss_iteration]
        if(key in loss_weights_table):
            return loss_weights_table[key]
        else:
            return 0.0

    def get_hyperparameter(self, key, default):
        hyperparams_table = self.hyper_param_config["hyperparams"][global_config.hyper_iteration]
        if (key in hyperparams_table):
            # print("Key ", key, " found. Returning ", global_config.loaded_network_config[key])
            return hyperparams_table[key]
        else:
            return default

    def get_all_hyperparams(self):
        return self.hyper_param_config["hyperparams"][global_config.hyper_iteration]

    def get_sr_version_name(self):
        network_version = global_config.sr_network_version
        hyper_iteration = global_config.hyper_iteration
        loss_iteration = global_config.loss_iteration

        return str(network_version) + "." + str(hyper_iteration) + "_" + str(loss_iteration)