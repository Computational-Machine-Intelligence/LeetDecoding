import setuptools

with open("README.md",'r',encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="leetDecoding",
    version = "0.0.2",
    keywords = ["pip", "leetDecoding","LeetDecoding","efficient_linear_decoding"],
    description = "Efficient computation library for linear attention.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "MIT Licence",
    url = "https://github.com/Computational-Machine-Intelligence/efficient_linear_decoding",
    
    author = "Jiaping Wang",
    author_email = "wjp666.s@gmail.com",
    packages=setuptools.find_packages(),
    # packages=setuptools.find_packages(
    #     where="efficient_linear_decoding",
    #     exclude=[
    #         "tests"
    #     ]
    # ),
    platforms = "any",
    install_requires=["pycuda", "triton==2.1.0","pynvml"], 
    include_package_data=True,
    # package_dir={"":"efficient_linear_decoding"},
    python_requires=">=3.8",
)