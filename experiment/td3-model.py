import os
import rclpy
from rclpy.node import Node
from collections import deque
from squaternion import Quaternion
import subprocess
import math
import threading
import time
import random
import logging
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# ROS and Gazebo message imports
from gazebo_msgs.msg import ModelState, ContactsState, EntityState, ModelStates
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam

# Global variables for sensor data
depth_image = None
imu_data = None
velocity_data = None
pose_data = None
collision_detected = None
reset_flag = False

class ReplayBuffer:
    """
    Experience replay buffer for reinforcement learning.
    Stores past experiences to sample from during training.
    """
    def __init__(self, max_size, image_shape, numerical_dim, action_dim):
        self.max_size = max_size
        self.current_size = 0
        self.pointer = 0

        # Initialize buffers for state, actions, rewards, and done flags
        self.state_image_buffer = np.zeros((max_size, *image_shape), dtype=np.float32)
        self.state_numerical_buffer = np.zeros((max_size, numerical_dim), dtype=np.float32)
        self.action_buffer = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state_image_buffer = np.zeros((max_size, *image_shape), dtype=np.float32)
        self.next_state_numerical_buffer = np.zeros((max_size, numerical_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((max_size,), dtype=np.float32)
        self.done_buffer = np.zeros((max_size,), dtype=np.float32)

    def store(self, state_image, state_numerical, action, next_state_image, next_state_numerical, reward, done):
        """
        Store a new experience in the replay buffer.
        """
        self.state_image_buffer[self.pointer] = state_image
        self.state_numerical_buffer[self.pointer] = state_numerical
        self.action_buffer[self.pointer] = action
        self.next_state_image_buffer[self.pointer] = next_state_image
        self.next_state_numerical_buffer[self.pointer] = next_state_numerical
        self.reward_buffer[self.pointer] = reward
        self.done_buffer[self.pointer] = done

        # Update pointer and size
        self.pointer = (self.pointer + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.
        """
        indices = np.random.randint(0, self.current_size, size=batch_size)
        return {
            "state_image": self.state_image_buffer[indices],
            "state_numerical": self.state_numerical_buffer[indices],
            "action": self.action_buffer[indices],
            "next_state_image": self.next_state_image_buffer[indices],
            "next_state_numerical": self.next_state_numerical_buffer[indices],
            "reward": self.reward_buffer[indices],
            "done": self.done_buffer[indices],
        }

    def get_size(self):
        """
        Get the current number of stored experiences.
        """
        return self.current_size

    def clear(self):
        """
        Clear the replay buffer.
        """
        self.pointer = 0
        self.current_size = 0
        self.state_image_buffer.fill(0)
        self.state_numerical_buffer.fill(0)
        self.action_buffer.fill(0)
        self.next_state_image_buffer.fill(0)
        self.next_state_numerical_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.done_buffer.fill(0)

class Actor(Model):

    def __init__(self, image_input_shape, numerical_input_dim, action_space_dim, max_action_value):
        super(Actor, self).__init__()
        self.max_action_value = max_action_value

        # Convolutional layers for image processing
        self.conv_layer1 = layers.Conv2D(32, kernel_size=3, strides=2, activation='relu')
        self.conv_layer2 = layers.Conv2D(64, kernel_size=3, strides=2, activation='relu')
        self.conv_layer3 = layers.Conv2D(128, kernel_size=3, strides=2, activation='relu')
        self.flatten_layer = layers.Flatten()

        # Fully connected layers for numerical input processing
        self.fc_layer1_numerical = layers.Dense(64, activation='relu')
        self.fc_layer2_numerical = layers.Dense(64, activation='relu')

        # Combined fully connected layers
        self.fc_layer1_combined = layers.Dense(256, activation='relu')
        self.fc_layer2_combined = layers.Dense(256, activation='relu')

        # Output layer for action prediction
        self.action_output_layer = layers.Dense(action_space_dim, activation='tanh')

    def call(self, inputs):
        image_input, numerical_input = inputs

        # Ensure image input is in the correct shape
        if len(image_input.shape) == 3:
            image_input = tf.expand_dims(image_input, axis=-1)

        # Process image input through convolutional layers
        image_features = self.conv_layer1(image_input)
        image_features = self.conv_layer2(image_features)
        image_features = self.conv_layer3(image_features)
        image_features = self.flatten_layer(image_features)

        # Process numerical input through fully connected layers
        numerical_features = self.fc_layer1_numerical(numerical_input)
        numerical_features = self.fc_layer2_numerical(numerical_features)

        # Concatenate the image and numerical features
        combined_features = layers.Concatenate()([image_features, numerical_features])

        # Process combined features through fully connected layers
        combined_features = self.fc_layer1_combined(combined_features)
        combined_features = self.fc_layer2_combined(combined_features)

        # Output the action and scale it
        predicted_action = self.action_output_layer(combined_features)
        scaled_action = tf.multiply(predicted_action, self.max_action_value)

        return scaled_action


class Critic(Model):

    def __init__(self, image_input_shape, numerical_input_dim, action_space_dim):
        super(Critic, self).__init__()

        # Convolutional layers for image processing
        self.conv_layer1 = layers.Conv2D(32, kernel_size=3, strides=2, activation='relu')
        self.conv_layer2 = layers.Conv2D(64, kernel_size=3, strides=2, activation='relu')
        self.conv_layer3 = layers.Conv2D(128, kernel_size=3, strides=2, activation='relu')
        self.flatten_layer = layers.Flatten()

        # Fully connected layers for numerical input processing
        self.fc_layer1_numerical = layers.Dense(64, activation='relu')
        self.fc_layer2_numerical = layers.Dense(64, activation='relu')

        # Input for action
        self.action_input_layer = layers.Input(shape=(action_space_dim,))

        # Combined fully connected layers
        self.fc_layer1_combined = layers.Dense(256, activation='relu')
        self.fc_layer2_combined = layers.Dense(256, activation='relu')

        # Q-value output layers
        self.q_value_output1 = layers.Dense(1)
        self.q_value_output2 = layers.Dense(1)

    def call(self, inputs):
        image_input, numerical_input, action_input = inputs

        # Ensure image input is in the correct shape
        if len(image_input.shape) == 3:
            image_input = tf.expand_dims(image_input, axis=-1)

        # Process image input through convolutional layers
        image_features = self.conv_layer1(image_input)
        image_features = self.conv_layer2(image_features)
        image_features = self.conv_layer3(image_features)
        image_features = self.flatten_layer(image_features)

        # Process numerical input through fully connected layers
        numerical_features = self.fc_layer1_numerical(numerical_input)
        numerical_features = self.fc_layer2_numerical(numerical_features)

        # Concatenate the image, numerical, and action inputs
        combined_features = layers.Concatenate()([image_features, numerical_features, action_input])

        # Process combined features through fully connected layers
        combined_features = self.fc_layer1_combined(combined_features)
        combined_features = self.fc_layer2_combined(combined_features)

        # Output the Q-values
        q_value1 = self.q_value_output1(combined_features)
        q_value2 = self.q_value_output2(combined_features)

        return q_value1, q_value2

class TD3(object):

    def __init__(self, state_image_shape, numerical_dim, action_dim, max_action_value, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, buffer_size=int(1e6), batch_size=64):
        
        self.actor = Actor(state_image_shape, numerical_dim, action_dim, max_action_value)
        self.critic = Critic(state_image_shape, numerical_dim, action_dim)
        
        self.target_actor = Actor(state_image_shape, numerical_dim, action_dim, max_action_value)
        self.target_critic = Critic(state_image_shape, numerical_dim, action_dim)

        # Initialize target networks with the weights of the respective models
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        # Hyperparameters
        self.max_action_value = max_action_value
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        
        # Replay buffer to store experiences
        self.replay_buffer = ReplayBuffer(buffer_size, state_image_shape, numerical_dim, action_dim)

    
    def update_target_weights(self, target_model, model, tau):
        """
        Update the target model weights using soft updates.
        """
        target_weights = target_model.get_weights()
        model_weights = model.get_weights()
        
        new_weights = [
            tau * model_weight + (1 - tau) * target_weight
            for target_weight, model_weight in zip(target_weights, model_weights)
        ]
        target_model.set_weights(new_weights)

    
    def add_noise(self, action, noise_std=0.1):
        """
        Add noise to the action for exploration.
        """
        noise = np.random.normal(0, noise_std, size=action.shape)
        noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        noisy_action = action + noise
        return np.clip(noisy_action, -self.max_action_value, self.max_action_value)


    def train(self, replay_buffer, iterations):
        """
        Train the TD3 agent.
        """
        total_critic_loss = 0
        total_actor_loss = 0
        avg_Q_value = 0
        max_Q_value = -float('inf')

        for it in range(iterations):
            # Sample a batch from the replay buffer
            batch = replay_buffer.sample(self.batch_size)
            state_image = tf.convert_to_tensor(batch['state_image'], dtype=tf.float32)
            state_numerical = tf.convert_to_tensor(batch['state_numerical'], dtype=tf.float32)
            action = tf.convert_to_tensor(batch['action'], dtype=tf.float32)
            reward = tf.convert_to_tensor(batch['reward'], dtype=tf.float32)
            done = tf.convert_to_tensor(batch['done'], dtype=tf.float32)
            next_state_image = tf.convert_to_tensor(batch['next_state_image'], dtype=tf.float32)
            next_state_numerical = tf.convert_to_tensor(batch['next_state_numerical'], dtype=tf.float32)

            # Get next action from target actor and add noise for exploration
            next_action = self.target_actor([next_state_image, next_state_numerical])
            noise = tf.random.normal(shape=tf.shape(next_action), mean=0.0, stddev=self.policy_noise)
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            next_action = tf.clip_by_value(next_action + noise, -self.max_action_value, self.max_action_value)

            # Get the target Q values from target critic
            target_Q1, target_Q2 = self.target_critic([next_state_image, next_state_numerical, next_action])
            target_Q = tf.minimum(target_Q1, target_Q2)

            # Bellman update for Q value
            target_Q = reward + (1 - done) * self.gamma * target_Q

            # Train the Critic
            with tf.GradientTape() as tape:
                current_Q1, current_Q2 = self.critic([state_image, state_numerical, action])
                critic_loss = tf.reduce_mean(self.mse_loss(current_Q1, target_Q)) + \
                              tf.reduce_mean(self.mse_loss(current_Q2, target_Q))

            # Apply gradients to the Critic
            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

            total_critic_loss += critic_loss
            avg_Q_value += tf.reduce_mean(target_Q)
            max_Q_value = max(max_Q_value, tf.reduce_max(target_Q))

            # Delayed policy update (actor and target networks)
            if it % self.policy_delay == 0:
                # Train the Actor (policy gradient)
                with tf.GradientTape() as tape:
                    predicted_action = self.actor([state_image, state_numerical])
                    actor_loss = -tf.reduce_mean(self.critic([state_image, state_numerical, predicted_action])[0])

                # Apply gradients to the Actor
                actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

                # Soft update for target actor and target critic
                self.update_target_weights(self.target_actor, self.actor, self.tau)
                self.update_target_weights(self.target_critic, self.critic, self.tau)

                total_actor_loss += actor_loss

        # Return average losses and Q values
        return total_critic_loss / iterations, avg_Q_value / iterations, max_Q_value, total_actor_loss / (iterations // self.policy_delay)


class GazeboEnv(Node):
  
    # Get binaries! https://packages.osrfoundation.org/gazebo/ubuntu/
    # RUN curl -L https://github.com/osrf/gazebo_models/archive/refs/heads/master.zip -o /tmp/gazebo_models.zip \
    # https://wiki.ros.org/tum_simulator
  
    def __init__(self):
        super().__init__('env')
        self.get_logger().info('Gazebo environment initialized!')

        self.done = False
        self.target = False
        self.state = {'image': None, 'numerical': None}

        self.set_self_state = self._initialize_model_state()

        self.depth_image = None
        self.imu_data = None
        self.velocity_data = None
        self.position_data = None
        self.collision = False

        self.vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 1)
        self.set_state = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.takeoff = self.create_publisher(std.Empty, '/simple_drone/takeoff', 10)
        self.land = self.create_publisher(std.Empty, '/simple_drone/land', 10)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")

        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        self.req = Empty.Request
        self.TIME_DELTA = 0.3
        self.rate = self.create_rate(1)

        self.goal = [0.0, 0.0, 0.0]
        self.obstacles = []
        self.bounds = [[-5.0, 5.0], [-5.0, 5.0]]
        self.max_height = 6.0

    def _initialize_model_state(self):
        model_state = ModelState()
        model_state.model_name = "r1"
        model_state.pose.position.x = 0.0
        model_state.pose.position.y = 0.0
        model_state.pose.position.z = 0.0
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = 0.0
        model_state.pose.orientation.w = 1.0
        return model_state

    def get_obs(self):
        return self.state

    def normalize_numerical_state(self, numerical_state):
        numerical_state = np.array(numerical_state)
        mean = np.mean(numerical_state, axis=0)
        std = np.std(numerical_state, axis=0)

        return (numerical_state - mean) / std

    def get_num_state(self):
        numerical_state = []
        temp = [self.imu_data, self.velocity_data, self.position_data]

        for obs in temp:
            for data in obs.values():
                numerical_state.extend(data)

        numerical_state = np.array(numerical_state)
        return self.normalize_numerical_state(numerical_state)

    def step(self, action, timestep):
        vel_cmd = Twist()
        vel_cmd.linear.x, vel_cmd.linear.y, vel_cmd.linear.z = action[:3]
        vel_cmd.angular.x, vel_cmd.angular.y, vel_cmd.angular.z = action[3:]

        self.vel_pub.publish(vel_cmd)
        self._unpause_physics()

        time.sleep(self.TIME_DELTA)

        self._pause_physics()

        self.state['image'] = self.depth_image
        self.state['numerical'] = self.get_num_state()

        dist_to_goal = np.linalg.norm(np.array(self.goal) - np.array(self.position_data['position']))
        reward = self.get_reward(dist_to_goal, action, timestep)
        target = dist_to_goal < 0.5
        done = self.collision or target

        return self.state, reward, done, target

    def _unpause_physics(self):
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        try:
            self.unpause.call_async(Empty.Request())
        except Exception:
            self.get_logger().error("/unpause_physics service call failed")

    def _pause_physics(self):
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        try:
            self.pause.call_async(Empty.Request())
        except rclpy.ServiceException:
            self.get_logger().error("/pause_physics service call failed")

    def _wait_for_service(self, client, service_name):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{service_name} service not available, waiting...')
        try:
            client.call_async(Empty.Request())
        except rclpy.ServiceException:
            self.get_logger().error(f"{service_name} service call failed")

    def _generate_random_position(self):
        x, y = 0, 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = True
        return x, y

    def delete_model(self, model_name):
        self._wait_for_service(self.delete_client, '/delete_entity')

        delete_request = DeleteEntity.Request()
        delete_request.name = model_name
        self.delete_client.call_async(delete_request)

    def spawn_cylinder(self, model_name, x, y, height):
        with open("/usr/share/sdformat9/1.7/cylinder_shape.sdf", "r") as f:
            model_xml = f.read()

        model_xml = model_xml.replace("<length>2.0</length>", f"<length>{height}</length>")

        self._wait_for_service(self.spawn_client, '/spawn_entity')

        spawn_request = SpawnEntity.Request()
        spawn_request.name = model_name
        spawn_request.xml = model_xml
        spawn_request.reference_frame = 'world'

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        spawn_request.initial_pose = pose

        self.spawn_client.call_async(spawn_request)

    def build_world(self):
        CYLINDER_COUNT = 20
        for i in range(CYLINDER_COUNT):
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            model_name = f"cylinder_{i}"

            self.delete_model(model_name)
            self.spawn_cylinder(model_name, x, y, self.max_height)
            self.obstacles.append([x, y])

    def set_goal(self):
        goal_ok = False
        while not goal_ok:
            goal = [
                float(random.randint(self.bounds[0][0], self.bounds[0][1])),
                float(random.randint(self.bounds[1][0], self.bounds[1][1])),
                float(random.randint(0, self.max_height - 1.0))
            ]
            if goal not in self.obstacles:
                self.goal = goal
                goal_ok = True

    def get_reward(self, dist_to_goal, action, timestep):
        reward = 0.0
        progress_reward_scale = 6.0
        speed_reward_scale = 0.05
        timestep_scale = 0.5
        proximity_penalty_scale = -1.5
        smooth_penalty_scale = -0.01
        safe_distance_threshold = 1.5
        min_distance = 2.0

        if dist_to_goal < 1.0:
            reward = 100.0
        elif self.collision or self.position_data['position'][2] > self.max_height:
            reward = -100.0
        else:
            progress_reward = progress_reward_scale * (1.0 / (dist_to_goal + 1e-6))
            speed_reward = speed_reward_scale * np.linalg.norm([action[0], action[1], action[2]])
            smooth_penalty = smooth_penalty_scale * np.linalg.norm(self.imu_data['linear_acceleration'])**2

            proximity_penalty = 0.0
            if min_distance < safe_distance_threshold:
                proximity_penalty = proximity_penalty_scale * (safe_distance_threshold - min_distance)

            height_penalty = -10 if self.position_data['position'][2] > self.max_height - 1.0 or self.position_data['position'][2] < 1.0 else 0

            reward = progress_reward

        return reward

class DepthSubscriber(Node):
    """
    Subscriber for depth image data. The depth image is processed and displayed.
    """
    def __init__(self):
        super().__init__('depth_subscriber')  # Initialize the node with the name 'depth_subscriber'
        # Create subscription to the depth image topic '/simple_drone/depth/depth/image_raw'
        self.subscription = self.create_subscription(
            Image, 
            '/simple_drone/depth/depth/image_raw', 
            self.depth_image_callback, 
            10)
        self.bridge = CvBridge()  # Initialize the CvBridge to convert ROS image messages to OpenCV format
        self.data = None  # Variable to store the processed depth data

    def preprocess(self, img, max_dist=70.0):
        """
        Preprocess the depth image by resizing and normalizing values.
        """
        self.data = cv2.resize(img, (160, 90))  # Resize image to (160, 90)

        # Copy the image and normalize depth values
        self.data = np.copy(self.data)
        finite_mask = np.isfinite(self.data)

        # Replace infinite values with 1.0 (indicating no depth)
        self.data[np.isinf(self.data)] = 1.0
        # Normalize finite values by max distance
        self.data[finite_mask] = self.data[finite_mask] / max_dist

        return self.data

    def depth_image_callback(self, msg):
        """
        Callback function for receiving depth image messages.
        """
        global depth
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = self.preprocess(cv_image)  # Preprocess the depth image

            # Display the processed depth image
            cv2.imshow("depth", depth)
            cv2.waitKey(3)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")


class ImuSubscriber(Node):
    """
    Subscriber for IMU (Inertial Measurement Unit) data.
    """
    def __init__(self):
        super().__init__('imu_subscriber')  # Initialize the node with the name 'imu_subscriber'
        # Create subscription to IMU data topic '/simple_drone/imu/out'
        self.subscription = self.create_subscription(
            Imu, 
            '/simple_drone/imu/out', 
            self.imu_callback, 
            10)
        self.data = {
            'orientation': [0.0, 0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0],
            'linear_acceleration': [0.0, 0.0, 0.0]
        }

    def imu_callback(self, msg):
        """
        Callback function for receiving IMU messages.
        """
        global imu
        # Store IMU data (orientation, angular velocity, and linear acceleration)
        self.data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }
        imu = self.data  # Update the global imu data


class VelocitySubscriber(Node):
    """
    Subscriber for velocity data.
    """
    def __init__(self):
        super().__init__('velocity_subscriber')  # Initialize the node with the name 'velocity_subscriber'
        # Create subscription to velocity topic '/simple_drone/gt_vel'
        self.subscription = self.create_subscription(
            Twist, 
            '/simple_drone/gt_vel', 
            self.velocity_callback, 
            1)
        self.data = {
            'linear_velocity': [0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0]
        }

    def velocity_callback(self, msg):
        """
        Callback function for receiving velocity messages.
        """
        global velocity
        # Store the linear and angular velocity data
        self.data = {
            'linear_velocity': [msg.linear.x, msg.linear.y, msg.linear.z],
            'angular_velocity': [msg.angular.x, msg.angular.y, msg.angular.z]
        }
        velocity = self.data  # Update the global velocity data


class CmdVelListener(Node):
    """
    Listener for receiving command velocity messages.
    """
    def __init__(self):
        super().__init__('cmd_vel_listener')  # Initialize the node with the name 'cmd_vel_listener'
        # Create subscription to cmd_vel topic '/simple_drone/cmd_vel'
        self.subscription = self.create_subscription(
            Twist, 
            '/simple_drone/cmd_vel', 
            self.cmd_vel_callback, 
            1)

    def cmd_vel_callback(self, msg):
        """
        Callback function for receiving command velocity messages.
        """
        # Log received command velocity values
        self.get_logger().info(f'Received cmd_vel: linear={msg.linear.x}, angular={msg.angular.z}')


class PoseSubscriber(Node):
    """
    Subscriber for pose data (position and orientation).
    """
    def __init__(self):
        super().__init__('pose_subscriber')  # Initialize the node with the name 'pose_subscriber'
        # Create subscription to pose topic '/simple_drone/gt_pose'
        self.subscription = self.create_subscription(
            Pose, 
            '/simple_drone/gt_pose', 
            self.pose_callback, 
            10)
        self.data = {
            'position': [0.0, 0.0, 0.0]
        }

    def pose_callback(self, msg):
        """
        Callback function for receiving pose messages.
        """
        global pose
        # Store position data
        position = msg.position
        self.data = {
            'position': [position.x, position.y, position.z]
        }
        pose = self.data  # Update the global pose data


class CollisionSubscriber(Node):
    """
    Subscriber for collision state data (bumper states).
    """
    def __init__(self):
        super().__init__('collision_subscriber')  # Initialize the node with the name 'collision_subscriber'
        # Create subscription to bumper states topic '/simple_drone/bumper_states'
        self.subscription = self.create_subscription(
            ContactsState, 
            '/simple_drone/bumper_states', 
            self.collision_callback, 
            10)
        self.data = None

    def collision_callback(self, msg):
        """
        Callback function for receiving collision state messages.
        """
        global collision
        # Check if there are any collision states
        if msg.states:
            self.data = True
            for state in msg.states:
                pass  # You can log collision details if necessary
        else:
            self.data = False
        collision = self.data  # Update the global collision state


def evaluate(network, epoch, eval_episodes=10, env=None):
    """
    Function to evaluate the performance of the agent after each epoch.
    It runs a number of evaluation episodes, collects the rewards and collision statistics.

    Parameters:
    - network: The neural network (agent) to evaluate.
    - epoch: The current epoch number.
    - eval_episodes: Number of episodes for evaluation.
    - env: The environment instance (GazeboEnv) for evaluation.

    Returns:
    - avg_reward: The average reward over all evaluation episodes.
    """
    avg_reward = 0.0  # Initialize total reward
    col = 0  # Initialize collision count

    for _ in range(eval_episodes):
        env.get_logger().info(f"Evaluating episode {_}")
        count = 0
        state = env.reset()  # Reset environment to initial state
        done = False

        while not done and count < 501:
            action = network.get_action(np.array(state))  # Get action from the network
            env.get_logger().info(f"Action: {action}")
            a_in = [(action[0] + 1) / 2, action[1]]  # Normalize action
            state, reward, done, _ = env.step(a_in)  # Take step in environment with action

            avg_reward += reward  # Accumulate reward
            count += 1
            if reward < -90:  # Check for collision or failure
                col += 1

    avg_reward /= eval_episodes  # Calculate average reward
    avg_col = col / eval_episodes  # Calculate collision rate

    # Log evaluation results
    env.get_logger().info("..............................................")
    env.get_logger().info(
        f"Average Reward over {eval_episodes} Evaluation Episodes, Epoch {epoch}: avg_reward {avg_reward:.6f}, avg_col {avg_col:.6f}"
    )
    env.get_logger().info("..............................................")

    return avg_reward  # Return the average reward


def main():
    """
    Main training loop for the reinforcement learning agent.
    It (tries to) initialize the ROS2 nodes, sets up the environment and agent, and runs training episodes.
    # TO FIX!
    """
    rclpy.init(args=None)  # Initialize ROS2

    timestep = 0
    epoch = 0
    timesteps_since_eval = 0
    episode_num = 0
    max_ep = # Maximum steps per episode
    eval_freq = # Frequency of evaluation
    max_timestep = # Maximum number of timesteps for training
    done = True
    numerical_state_dim = # Dimensionality of the numerical state
    action_dim = # Dimensionality of the action space
    max_action = # Maximum action value
    depth_img_shape = # Shape of the depth image
    evaluations = []

    expl_noise = # Initial exploration noise
    expl_min = # Minimum exploration noise
    expl_decay_steps = # Number of steps to decay exploration noise

    # Initialize ROS2 subscribers
    depth_sub = DepthSubscriber()
    imu_sub = ImuSubscriber()
    velocity_sub = VelocitySubscriber()
    pose_sub = PoseSubscriber()
    collision_sub = CollisionSubscriber()
    cmd_vel_sub = CmdVelListener()

    # Executor to handle multiple ROS2 nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(depth_sub)
    executor.add_node(imu_sub)
    executor.add_node(velocity_sub)
    executor.add_node(pose_sub)
    executor.add_node(collision_sub)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Initialize the neural network, environment, and replay buffer
    network = TD3(depth_img_shape, numerical_state_dim, action_dim, max_action)
    env = GazeboEnv()  # Initialize the environment
    replay_buffer = ReplayBuffer(max_size=1e6, image_shape=depth_img_shape, numerical_dim=numerical_state_dim, action_dim=action_dim)

    try:
        while rclpy.ok() and timestep < max_timestep:
            if done:
                env.get_logger().info(f'Episode {episode_num} done. Timestep: {timestep}')

                if timestep != 0:
                    env.get_logger().info(f'\n\nReplay buffer: {replay_buffer.reward_buffer}\n\nTraining at timestep {timestep}')
                    network.train(replay_buffer, episode_timesteps)

                if timesteps_since_eval >= eval_freq:
                    env.get_logger().info(f'Validating at epoch {epoch}')
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=10, env=env))

                    epoch += 1

                # Reset environment for the next episode
                state = env.reset()
                env.goal_viz.destroy_node()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            env.get_obs()
            env.goal_viz.publish_marker()  # Visualize goal

            # Decay exploration noise
            if expl_noise > expl_min:
                expl_noise -= ((expl_noise - expl_min) / expl_decay_steps)

            # Select action using the actor network
            action = network.actor((
                np.expand_dims(np.asarray(state['image']).astype(np.float32), axis=0),
                np.expand_dims(np.asarray(state['numerical']).astype(np.float32), axis=0)
            ))[0].numpy()
            # Add noise for exploration
            action = network.add_noise(action, noise_std=expl_noise)
            # Take a step in the environment
            next_state, reward, done, target = env.step(action, episode_timesteps)

            if episode_timesteps + 1 == max_ep:
                done = True

            done_bool = float(done)
            # Log the episode reward
            env.get_logger().info(f'\nEpisode reward: {episode_reward}')
            # Store the transition in the replay buffer
            replay_buffer.store(
                state['image'], state['numerical'], action,
                next_state['image'], next_state['numerical'], reward, done_bool
            )

            state = next_state
            episode_reward += reward
            episode_timesteps += 1
            timestep += 1
            timesteps_since_eval += 1

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
