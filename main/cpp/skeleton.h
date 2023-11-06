// %BANNER_BEGIN%
// ---------------------------------------------------------------------
// %COPYRIGHT_BEGIN%
// Copyright (c) 2022 Magic Leap, Inc. All Rights Reserved.
// Use of this file is governed by the Software License Agreement,
// located here: https://www.magicleap.com/software-license-agreement-ml2
// Terms and conditions applicable to third-party materials accompanying
// this distribution may also be found in the top-level NOTICE file
// appearing herein.
// %COPYRIGHT_END%
// ---------------------------------------------------------------------
// %BANNER_END%

#include <array>

#include <app_framework/application.h>
#include <ml_hand_tracking.h>

#define GLM_ENABLE_EXPERIMENTAL 1
#include <glm/gtx/transform.hpp>

constexpr glm::vec4 KEYPOINT_COLOR_THUMB = glm::vec4(1.f, 0.f, 0.f, 1.f);
constexpr glm::vec4 KEYPOINT_COLOR_INDEX = glm::vec4(0.f, 1.f, 0.f, 1.f);
constexpr glm::vec4 KEYPOINT_COLOR_MIDDLE = glm::vec4(1.f, 1.f, 1.f, 1.f);
constexpr glm::vec4 KEYPOINT_COLOR_RING = glm::vec4(0.f, 0.f, 1.f, 1.f);
constexpr glm::vec4 KEYPOINT_COLOR_PINKY = glm::vec4(1.f, 1.f, 0.f, 1.f);
constexpr glm::vec4 KEYPOINT_COLOR_WRIST = glm::vec4(1.f, 0.f, 1.f, 1.f);
constexpr glm::vec4 KEYPOINT_COLOR_CENTER = glm::vec4(0.f, 1.f, 1.f, 1.f);

class FingerSkeleton {
public:
  // Initialize the finger mesh and add it to the given root node
  void Initialize(std::shared_ptr<ml::app_framework::Node> root, const glm::vec4& color = {1.0f, 1.0f, 1.0f, 1.0f});

  /** Update finger mesh based on the given vertices
   * @param cmc Carpals-Meta-carpal vertex
   * @param mcp Meta-carpal phalengial vertex
   * @param ip Inter-phalengial vertex
   * @param tip Tip of finger vertex
   **/
  void Update(const glm::vec3 &cmc, const glm::vec3 &mcp, const glm::vec3 &ip, const glm::vec3 &tip);

  // Hide mesh
  void Hide();

  // Enable/Disable bone coloring
  void SetColorOverride(bool draw_color);

private:
  std::shared_ptr<ml::app_framework::Node> bone_;
};

class HandSkeleton {
public:
  // Initialize the hand mesh and add it to the given root node
  void Initialize(std::shared_ptr<ml::app_framework::Node> root);

  // Update hand mesh based on the given keypoints
  void Update(const std::array<glm::vec3, MLHandTrackingStaticData_MaxKeyPoints> &keypoints, const bool is_visible);

  // Enable/Disable bone coloring
  void SetColorOverride(bool draw_color);

private:
  FingerSkeleton thumb_;
  FingerSkeleton index_;
  FingerSkeleton middle_;
  FingerSkeleton ring_;
  FingerSkeleton pinky_;
};