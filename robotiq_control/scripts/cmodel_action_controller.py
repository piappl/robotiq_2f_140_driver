#!/usr/bin/env python3
import rospy, os
import numpy as np
from sensor_msgs.msg import JointState
# Actionlib
from actionlib import SimpleActionServer
from robotiq_msgs.msg import (
    CModelCommand,
    CModelStatus,
    CModelCommandAction,
    CModelCommandFeedback,
    CModelCommandResult,
)
from control_msgs.msg import (
    GripperCommandAction,
    GripperCommandFeedback,
    GripperCommandResult
)


def read_parameter(name, default):
  if not rospy.has_param(name):
    rospy.logwarn('Parameter [%s] not found, using default: %s' % (name, default))
  return rospy.get_param(name, default)

class CModelActionController(object):
  def __init__(self, activate=True):
    self._ns = rospy.get_namespace()
    controller_name = "gripper_action_controller"
    # Read configuration parameters
    self._fb_rate = read_parameter(self._ns + controller_name + '/publish_rate', 60.0)
    self._min_gap_counts = read_parameter(self._ns + controller_name + '/min_gap_counts', 230.)
    self._counts_to_meters = read_parameter(self._ns + controller_name + '/counts_to_meters', 0.8)
    self._position = 0.0
    self._min_gap = read_parameter(self._ns + controller_name + '/min_gap', 0.0)
    self._max_gap = read_parameter(self._ns + controller_name + '/max_gap', 0.085)
    self._speed = read_parameter(self._ns + controller_name + '/speed', 0.5)
    self._min_speed = read_parameter(self._ns + controller_name + '/min_speed', 0.013)
    self._max_speed = read_parameter(self._ns + controller_name + '/max_speed', 0.1)
    self._force = read_parameter(self._ns + controller_name + '/force', 60)
    self._min_force = read_parameter(self._ns + controller_name + '/min_force', 40.0)
    self._max_force = read_parameter(self._ns + controller_name + '/max_force', 100.0)
    self._joint_name = read_parameter(self._ns + controller_name + '/joint_name', 'robotiq_85_left_knuckle_joint')
    self._gripper_prefix = read_parameter(self._ns + 'gripper_prefix', "")   # Used for updating joint state
    # Configure and start the action server
    self._status = CModelStatus()
    self._name = self._ns + controller_name
    self._server = SimpleActionServer(self._name, CModelCommandAction, execute_cb=self._execute_cb, auto_start = False)

    self._moveit_server = SimpleActionServer(self._ns + "moveit_action_controller", GripperCommandAction, execute_cb=self._moveit_execute_cb, auto_start = False)

    self.status_pub = rospy.Publisher('gripper_status', CModelCommandFeedback, queue_size=1)
    self.js_pub = rospy.Publisher('joint_states', JointState, queue_size=1)
    self.js_pub_global = rospy.Publisher('/joint_states', JointState, queue_size=1)
    rospy.Subscriber('status', CModelStatus, self._status_cb, queue_size=1)
    self._cmd_pub = rospy.Publisher('command', CModelCommand, queue_size=1)
    working = True
    rospy.sleep(1.0)   # Wait before checking status with self._ready()
    if activate and not self._ready():
      rospy.sleep(2.0)
      working = self._activate()
    if not working:
      return
    self._server.start()
    self._moveit_server.start()
    rospy.logdebug('%s: Started' % self._name)

  def _preempt(self):
    self._stop()
    rospy.loginfo('%s: Preempted' % self._name)
    self._server.set_preempted()
    self._moveit_server.set_preempted()

  def _status_cb(self, msg):
    self._status = msg
    # Publish the joint_states for the gripper
    js_msg = JointState()
    js_msg.header.stamp = rospy.Time.now()
    js_msg.name.append(self._joint_name)
    # js_msg.position.append(0.8*self._status.gPO/self._min_gap_counts)
    js_msg.position.append(self._counts_to_meters*
                           self._status.gPO/self._min_gap_counts)
    self.js_pub.publish(js_msg)
    js_msg.name = []
    js_msg.name.append(self._gripper_prefix + self._joint_name)
    self.js_pub_global.publish(js_msg)

    # Publish the gripper status (to easily access gripper width)
    feedback = CModelCommandFeedback()
    feedback.activated = self._ready()
    feedback.position = self._get_position()
    feedback.stalled = self._stalled()
    # # feedback.reached_goal = self._reached_goal(position)
    self.status_pub.publish(feedback)

  def _execute_cb(self, goal):
    # Check that the gripper is active. If not, activate it.
    if not self._check_active():
      return
    # check that preempt has not been requested by the client
    if self._server.is_preempt_requested():
      self._preempt()
      return
    # Clip the goal
    position = np.clip(goal.position, self._min_gap, self._max_gap)
    velocity = np.clip(goal.velocity, self._min_speed, self._max_speed)
    force = np.clip(goal.force, self._min_force, self._max_force)
    # Send the goal to the gripper and feedback to the action client
    rate = rospy.Rate(self._fb_rate)
    rospy.logdebug('%s: Moving gripper to position: %.3f ' % (self._name, position))

    self._status.gOBJ = 0 # R.Hanai

    feedback = CModelCommandFeedback()

    command_sent_time = rospy.get_rostime()
    while not self._reached_goal(position):
      self._goto_position(position, velocity, force)
      if rospy.is_shutdown() or self._server.is_preempt_requested():
        self._preempt()
        return
      feedback.position = self._get_position()
      feedback.stalled = self._stalled()
      feedback.reached_goal = self._reached_goal(position)
      self._server.publish_feedback(feedback)
      rate.sleep()

      time_since_command = rospy.get_rostime() - command_sent_time
      if time_since_command > rospy.Duration(0.5) and self._stalled():
        break
    rospy.logdebug('%s: Succeeded' % self._name)
    result = CModelCommandResult()
    result.position = self._get_position()
    result.stalled = self._stalled()
    result.reached_goal = self._reached_goal(position)
    self._server.set_succeeded(result)

  def _moveit_execute_cb(self, goal):
    # Check gripper is active
    if not self._check_active():
      return
    # Check preemption
    if self._moveit_server.is_preempt_requested():
      self._preempt()
      return
    # Clip goal
    rospy.logdebug('%s: Goal position: %.3f' % (self._name, goal.command.position))
    self._position = np.clip(self._max_gap - goal.command.position, self._min_gap, self._max_gap)
    self._force = np.clip(goal.command.max_effort, self._min_force, self._max_force)
    # Feedback
    rate = rospy.Rate(self._fb_rate)
    rospy.logdebug('%s: Moving gripper to position: %.3f ' % (self._name, self._position))

    self._status.gOBJ = 0 # R.Hanai
    feedback = GripperCommandFeedback()
    command_sent_time = rospy.get_rostime()
    # Wait to Move
    while not self._reached_goal(self._position):
      self._goto_position(self._position, self._speed, self._force)
      if rospy.is_shutdown() or self._moveit_server.is_preempt_requested():
        self._preempt()
        return
      feedback.position = self._get_position()
      feedback.stalled = self._stalled()
      feedback.reached_goal = self._reached_goal(self._position)
      self._moveit_server.publish_feedback(feedback)
      rate.sleep()

      time_since_command = rospy.get_rostime() - command_sent_time
      if time_since_command > rospy.Duration(0.5) and self._stalled():
        break
    # Publish Result
    result = GripperCommandResult()
    result.position = self._get_position()
    result.effort = self._force
    result.stalled = self._stalled()
    result.reached_goal = self._reached_goal(self._position)
    rospy.logdebug("Reached Goal: %s "% self._reached_goal(self._position))
    self._moveit_server.set_succeeded(result)

  def _check_active(self):
    if not self._ready():
      if not self._activate():
        rospy.logwarn('%s could not accept goal because the gripper is not yet active' % self._name)
        return False
    return True

  def _activate(self, timeout=5.0):
    command = CModelCommand()
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150
    start_time = rospy.get_time()
    while not self._ready():
      if rospy.is_shutdown():
        self._preempt()
        return False
      if rospy.get_time() - start_time > timeout:
        rospy.logwarn('Failed to activate gripper in ns [%s]' % (self._ns))
        return False
      self._cmd_pub.publish(command)
      rospy.sleep(0.1)
    rospy.loginfo('Successfully activated gripper in ns [%s]' % (self._ns))
    return True

  def _get_position(self):
    gPO = self._status.gPO
    pos = np.clip((self._max_gap - self._min_gap)/(-self._min_gap_counts)*(gPO-self._min_gap_counts), self._min_gap, self._max_gap)
    return pos

  def _goto_position(self, pos, vel, force):
    """
    Goto position with desired force and velocity
    @type  pos: float
    @param pos: Gripper width in meters
    @type  vel: float
    @param vel: Gripper speed in m/s
    @type  force: float
    @param force: Gripper force in N
    """
    command = CModelCommand()
    command.rACT = 1
    command.rGTO = 1
    command.rPR = int(np.clip((-self._min_gap_counts)/(self._max_gap - self._min_gap) * (pos - self._min_gap) + self._min_gap_counts, 0, self._min_gap_counts))
    command.rSP = int(np.clip((255)/(self._max_speed - self._min_speed) * (vel - self._min_speed), 0, 255))
    command.rFR = int(np.clip((255)/(self._max_force - self._min_force) * (force - self._min_force), 0, 255))
    self._cmd_pub.publish(command)

  def _moving(self):
    return self._status.gGTO == 1 and self._status.gOBJ == 0

  def _reached_goal(self, goal, tol = 0.003):
    return (abs(goal - self._get_position()) < tol)

  def _ready(self):
    return self._status.gSTA == 3 and self._status.gACT == 1

  def _stalled(self):
    return self._status.gOBJ == 1 or self._status.gOBJ == 2

  def _stop(self):
    command = CModelCommand()
    command.rACT = 1
    command.rGTO = 0
    self._cmd_pub.publish(command)
    rospy.logdebug('Stopping gripper in ns [%s]' % (self._ns))


if __name__ == '__main__':
  node_name = os.path.splitext(os.path.basename(__file__))[0]
  rospy.init_node(node_name)
  cmodel_server = CModelActionController()
  rospy.spin()
