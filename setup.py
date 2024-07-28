from setuptools import find_packages,setup


setup(
    name="efficient_linear_decoding",
    packages=find_packages(
        exclude=[
            "tests"
        ]
    ),
    install_requires=["torch", "einops", "triton"], # 定义包的依赖
    version="0.0.5", # 设置包的版本号
    author="jiapingWang",
    include_package_data=True, # 指定是否包含包数据
)