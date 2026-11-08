def create_vm_manager_and_provider(provider_name: str, region: str, use_proxy: bool = False):
    """
    Factory function to get the Virtual Machine Manager and Provider instances based on the provided provider name.
    
    Args:
        provider_name (str): The name of the provider (e.g., "aws", "vmware", etc.)
        region (str): The region for the provider
        use_proxy (bool): Whether to use proxy-enabled providers (currently only supported for AWS)
    """
    provider_name = provider_name.lower().strip()
    if provider_name == "vmware":
        from onpolicy_env.desktop_env.providers.vmware.manager import VMwareVMManager
        from onpolicy_env.desktop_env.providers.vmware.provider import VMwareProvider
        return VMwareVMManager(), VMwareProvider(region)
    elif provider_name == "virtualbox":
        from onpolicy_env.desktop_env.providers.virtualbox.manager import VirtualBoxVMManager
        from onpolicy_env.desktop_env.providers.virtualbox.provider import VirtualBoxProvider
        return VirtualBoxVMManager(), VirtualBoxProvider(region)
    elif provider_name in ["aws", "amazon web services"]:
        from onpolicy_env.desktop_env.providers.aws.manager import AWSVMManager
        from onpolicy_env.desktop_env.providers.aws.provider import AWSProvider
        return AWSVMManager(), AWSProvider(region)
    elif provider_name == "azure":
        from onpolicy_env.desktop_env.providers.azure.manager import AzureVMManager
        from onpolicy_env.desktop_env.providers.azure.provider import AzureProvider
        return AzureVMManager(), AzureProvider(region)
    elif provider_name == "docker":
        from onpolicy_env.desktop_env.providers.docker.manager import DockerVMManager
        from onpolicy_env.desktop_env.providers.docker.provider import DockerProvider
        return DockerVMManager(), DockerProvider(region)
    elif provider_name == "aliyun":
        from onpolicy_env.desktop_env.providers.aliyun.manager import AliyunVMManager
        from onpolicy_env.desktop_env.providers.aliyun.provider import AliyunProvider
        return AliyunVMManager(), AliyunProvider()
    elif provider_name == "volcengine":
        from onpolicy_env.desktop_env.providers.volcengine.manager import VolcengineVMManager
        from onpolicy_env.desktop_env.providers.volcengine.provider import VolcengineProvider
        return VolcengineVMManager(), VolcengineProvider()
    elif provider_name == "fastvm":
        from onpolicy_env.desktop_env.providers.fastvm.manager import FastvmVMManager
        from onpolicy_env.desktop_env.providers.fastvm.provider import FastvmProvider
        return FastvmVMManager(), FastvmProvider(region)
    else:
        raise NotImplementedError(f"{provider_name} not implemented!")
