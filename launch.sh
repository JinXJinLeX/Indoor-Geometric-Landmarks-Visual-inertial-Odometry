gnome-terminal -t "title" -x bash -c "roslaunch startup start.launch;exrc bash"
#!/bin/bashsource ~/Documents/ros/devel/steup.bash
source ~/xjl_work_space/gvins_yolo_ws/devel/setup.bash
{
gnome-terminal -t "start_robot" -x bash -c "roslaunch vins_estimator parking.launch;exec bash"

}&sleep 1s
source ~/xjl_work_space/gvins_yolo_ws/devel/setup.bash
{
gnome-terminal -t "start_trx" -x bash -c "roslaunch vins_estimator third_rviz.launch;exec bash"
}
