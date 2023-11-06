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

#include "skeleton.h"

#include <app_framework/toolset.h>

#include <glm/gtc/quaternion.hpp>

void FingerSkeleton::Initialize(std::shared_ptr<ml::app_framework::Node> root, const glm::vec4& color) {
  bone_ = ml::app_framework::CreatePresetNode(ml::app_framework::NodeType::Line);
  root->AddChild(bone_);

  auto line = bone_->GetComponent<ml::app_framework::RenderableComponent>();
  line->SetVisible(true);
  line->GetMesh()->SetPrimitiveType(GL_LINE_STRIP);

  auto line_material = std::static_pointer_cast<ml::app_framework::FlatMaterial>(line->GetMaterial());
  line_material->SetColor(color);
  line_material->SetOverrideVertexColor(false);
}

void FingerSkeleton::Update(const glm::vec3 &cmc, const glm::vec3 &mcp, const glm::vec3 &ip, const glm::vec3 &tip) {
  constexpr size_t num_vertices = 4;
  const glm::vec3 vertices[num_vertices] = {cmc, mcp, ip, tip};
  auto line = bone_->GetComponent<ml::app_framework::RenderableComponent>();
  line->SetVisible(true);
  line->GetMesh()->UpdateMesh(vertices, nullptr, num_vertices, nullptr, 0);
}

void FingerSkeleton::Hide() {
  bone_->GetComponent<ml::app_framework::RenderableComponent>()->SetVisible(false);
}

void FingerSkeleton::SetColorOverride(bool draw_color) {
  auto line = bone_->GetComponent<ml::app_framework::RenderableComponent>();
  auto line_material = std::static_pointer_cast<ml::app_framework::FlatMaterial>(line->GetMaterial());
  line_material->SetOverrideVertexColor(draw_color);
}


void HandSkeleton::Initialize(std::shared_ptr<ml::app_framework::Node> root) {
  thumb_.Initialize(root, KEYPOINT_COLOR_THUMB);
  index_.Initialize(root, KEYPOINT_COLOR_INDEX);
  middle_.Initialize(root, KEYPOINT_COLOR_MIDDLE);
  ring_.Initialize(root, KEYPOINT_COLOR_RING);
  pinky_.Initialize(root, KEYPOINT_COLOR_PINKY);
}

void HandSkeleton::Update(const std::array<glm::vec3, MLHandTrackingStaticData_MaxKeyPoints> &keypoints,
                          const bool is_visible) {
  if (is_visible) {
#define GET_THUMB_VERTICES(finger)                                                                    \
  keypoints[MLHandTrackingKeyPoint_##finger##_CMC], keypoints[MLHandTrackingKeyPoint_##finger##_MCP], \
      keypoints[MLHandTrackingKeyPoint_##finger##_IP], keypoints[MLHandTrackingKeyPoint_##finger##_Tip]

#define GET_FINGER_VERTICES(finger)                                                                   \
  keypoints[MLHandTrackingKeyPoint_##finger##_MCP], keypoints[MLHandTrackingKeyPoint_##finger##_PIP], \
      keypoints[MLHandTrackingKeyPoint_##finger##_DIP], keypoints[MLHandTrackingKeyPoint_##finger##_Tip]

    thumb_.Update(GET_THUMB_VERTICES(Thumb));
    index_.Update(GET_FINGER_VERTICES(Index));
    middle_.Update(GET_FINGER_VERTICES(Middle));
    ring_.Update(GET_FINGER_VERTICES(Ring));
    pinky_.Update(GET_FINGER_VERTICES(Pinky));
  } else {
    thumb_.Hide();
    index_.Hide();
    middle_.Hide();
    ring_.Hide();
    pinky_.Hide();
  }
}

void HandSkeleton::SetColorOverride(bool draw_color) {
    thumb_.SetColorOverride(draw_color);
    index_.SetColorOverride(draw_color);
    middle_.SetColorOverride(draw_color);
    ring_.SetColorOverride(draw_color);
    pinky_.SetColorOverride(draw_color);
}
