#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}

std::string sign_id_to_class(const double input) {
    // 确定类别前缀
    std::string prefix;
    // 计算实际的数字部分（1-9）
    int number = static_cast<int>(input) % 10;

    // 根据输入数字的范围确定前缀
    if (input >= 11.0 && input <= 19.0) {
        prefix = "cir";
    } else if (input >= 21.0 && input <= 29.0) {
        prefix = "tri";
    } else if (input >= 31.0 && input <= 39.0) {
        prefix = "rect";
    } else {
       printf("Input number is out of expected range");
    }

    // 构造并返回结果字符串
    return prefix + "_" + std::to_string(number*10);
}

double sign_class_to_id(const std::string input)
{
 size_t underscore_pos = input.find('_');
    if (underscore_pos == std::string::npos) {
        printf("Input does not contain an underscore");
    }
    std::string prefix = input.substr(0, underscore_pos);

    // 提取数字部分并转换为整数
    std::string number_str = input.substr(underscore_pos + 1);
    int number = std::stoi(number_str);

    // 根据前缀和数字生成对应的输出值
    if (prefix == "cir") {
        return 10 + number; // 11 to 19
    } else if (prefix == "tri") {
        return 20 + number; // 21 to 29
    } else if (prefix == "rect") {
        return 30 + number; // 31 to 39
    } else {
        printf("Unknown prefix");
    }
}