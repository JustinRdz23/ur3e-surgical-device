from setuptools import find_packages, setup

package_name = 'dual_servo_ur3e_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='justinrc',
    maintainer_email='A01611412@tec.mx',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['ur2mujoco_bridge = dual_servo_ur3e_sim.mujoco_node:main'
        ],
    },
)
