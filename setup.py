import setuptools


def read_file(name):
    with open(name, "r", encoding="UTF-8") as f:
        return f.read().strip()


install_requires = read_file("./config/install_requires.txt").split('\n')
project_name = read_file("./config/project_name.txt")
version = __import__(project_name).__version__
long_description = read_file('./README.md')


setuptools.setup(
    name=project_name,
    version=version,
    author="genggui001",
    author_email="genggui001@gmail.com",
    description="Pytorch lightning help",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/genggui001/{project_name}",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
)
