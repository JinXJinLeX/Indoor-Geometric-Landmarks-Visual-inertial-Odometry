#include "map_manager.h"

void LocalMapManager::clearsignState()
{
    sign.clear();
}

LocalMapManager::LocalMapManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void LocalMapManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

int LocalMapManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : sign)
    {

        it.used_num = it.sign_per_frame.size();

        if (it.used_num != 0)
        {
            cnt++;
        }
    }
    return cnt;
}

void LocalMapManager::removesignBack()
{
    for (auto it = sign.begin(), it_next = sign.begin(); // 遍历滑窗标志
         it != sign.end(); it = it_next)
    {
        if (it->used_num != 0)
        {
            for (int i = 0; i < WINDOW_SIZE - 1; i++)
            {
                it->is_detected[i] = it->is_detected[i + 1];
            }
            it->is_detected[WINDOW_SIZE] = 0;
        }
        else
        {
            it->sign_per_frame.erase(it->sign_per_frame.begin());
            if (it->sign_per_frame.size() == 0)
                sign.erase(it);
        }
    }
}

void LocalMapManager::removesignFront()
{
    for (auto it = sign.begin(), it_next = sign.begin(); it != sign.end(); it = it_next) // 遍历滑窗标志
    {
        it_next++;

        if (it->used_num != 0 && it->is_detected[WINDOW_SIZE] != 1)
        {
            ;
        }
        else
        {
            it->sign_per_frame.erase(it->sign_per_frame.begin());
            if (it->used_num == 0)
                sign.erase(it);
        }
    }
}

void LocalMapManager::removesignOutlier()
{
    // ROS_BREAK();
    // int i = -1;
    // for (auto it = sign.begin(), it_next = sign.begin();
    //      it != sign.end(); it = it_next)
    // {
    //     it_next++;
    //     i += it->used_num != 0;
    //     if (it->used_num != 0 && it->is_outlier == true)
    //     {
    //         sign.erase(it);
    //     }
    // }
}

void LocalMapManager::addSignCheck(Vector3d C, string c, int &id)
{
    // bool isfound;
    // int is_detected[WINDOW_SIZE];
    if (!sign.empty())
    {
        for (auto sign_ : sign)
        {
            if (((sign_.C_ - C).norm() < 2) && sign_.classify == c)
            {
                // isfound = true;
                id = sign_.sign_id;
                // cout<<sign_.C_<<endl;
                // sign_.used_num = sign_.used_num + 1;
            }
        }
    }
    return;
}

void LocalMapManager::initialSign(int &id, string c, Vector3d C, Vector3d N, double time_, vector<Vector2d> pc, int flag)
{
    // int is_detected[WINDOW_SIZE];
    // is_detected[WINDOW_SIZE - 1] = 1;
    // if (id == -1 && flag == 0) // 管理器、地图中都没有
    SignPerFrame tempfra(pc, time_);
    SignPerId tempsign(id, c, C, N, 1);
    if (flag == -1) // 初始化，把地图中的标志读入管理器
    {
        tempsign.sign_per_frame.clear();
        sign.push_back(tempsign);

    }
    if (flag == 0) // 管理器、地图中都没有
    {
        // id = sign.size()+1;
        tempsign.is_detected[WINDOW_SIZE - 1] = 1;
        tempsign.sign_per_frame.push_back(tempfra);
        sign.push_back(tempsign);
    }
    else
    {
        if (flag == 1) // 管理器中有
        {
            for (auto it = sign.begin(); it != sign.end(); it++)
            {
                if (it->sign_id == id)
                {
                    it->is_detected[WINDOW_SIZE - 1] = 0;
                    // SignPerFrame sign_per_fra(pc, time_);
                    it->sign_per_frame.push_back(tempfra);
                }
            }

        }
        if (flag == 2) // 管理器中没有但地图中有
        {

            tempsign.is_detected[WINDOW_SIZE - 1] = 1;
            tempsign.sign_per_frame.push_back(tempfra);
            sign.push_back(tempsign);
        }
    }
}

void LocalMapManager::debugShowsign()
{
    ROS_DEBUG("debug show");
    for (auto &it : sign)
    {
        ROS_ASSERT(it.sign_per_frame.size() != 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.sign_id, it.used_num);
        int sum = 0;
        for (auto &j : it.sign_per_frame)
        {
            ROS_DEBUG("%d,", int(j.time));
            sum += 1;
            printf("(%lf,%lf) ", j.pts[0].x(), j.pts[0].y());
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

void LocalMapManager::updateSign(double *time_)
{
    if (!sign.empty())
    {
        for (auto it = sign.begin(), it_next = sign.begin(); it != sign.end(); it = it_next)
        {
            it_next++;
            for (int j = 0; j < WINDOW_SIZE; j++)
            {
                it->is_detected[j] = 0;
            }
            while (!it->sign_per_frame.empty() && it->sign_per_frame.front().time < time_[0])
            {
                it->sign_per_frame.erase(it->sign_per_frame.begin());
            }
            int num = 0;
            for (auto per_frame : it->sign_per_frame)
            {
                int i = 0;
                while (i < WINDOW_SIZE)
                {
                    if (time_[i] == per_frame.time)
                    {
                        it->is_detected[i] = 1;
                        num += 1;
                    }
                    i++;
                }
            }
            it->used_num = num;
            // cout << "used num is:" << it->used_num << endl;
            if (num == 0)
            {
                sign.erase(it);
            }
        }
    }
}

int LocalMapManager::getSignCount()
{
    int cnt = 0;
    for (auto &it : sign)
    {
        it.used_num = it.sign_per_frame.size();
        cout << "used num is:" << it.used_num << endl;
        if (it.used_num >= 2)
        {
            cnt++;
        }
    }
    return cnt;
}

void LocalMapManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = sign.begin(), it_next = sign.begin();
         it != sign.end(); it = it_next)
    {
        it_next++;

        if (it->used_num != 0)
            it->used_num--;
        else
        {
            Eigen::Vector3d uv_i = it->sign_per_frame[0].N;
            it->sign_per_frame.erase(it->sign_per_frame.begin());
            if (it->sign_per_frame.size() < 2)
            {
                sign.erase(it);
                continue;
            }
            else
            {
                // Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                // Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                // Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                // double dep_j = pts_j(2);
                // if (dep_j > 0)
                //     it->estimated_depth = dep_j;
                // else
                //     it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}