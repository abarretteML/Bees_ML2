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

#define ALOG_TAG "com.magicleap.capi.sample.hand_tracking"

#include "skeleton.h"

#include <array>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <iostream>

#include <app_framework/application.h>
#include <app_framework/convert.h>
#include <app_framework/gui.h>
#include <app_framework/toolset.h>
#include <ml_hand_tracking.h>
#include <ml_perception.h>

#include <imgui.h>

namespace {
constexpr glm::vec3 KEYPOINT_CUBE_SCALE = glm::vec3(.01f, .01f, .01f);
constexpr glm::vec3 PARTICLE_SCALE = glm::vec3(.001f, .001f, .001f);
constexpr glm::vec3 KEYPOINT_LABEL_SCALE = glm::vec3(.000625f, -.000625f, 1.f);
const double DRAG = .98;
const double FORCE_SCALE = 1;
const int N_PARTICLES = 100;
const float_t SPACE_SCALE = 10;
float DT = .1;

constexpr std::array<glm::vec4, MLHandTrackingStaticData_MaxKeyPoints> KEYPOINT_COLORS {{
  KEYPOINT_COLOR_THUMB, KEYPOINT_COLOR_THUMB, KEYPOINT_COLOR_THUMB, KEYPOINT_COLOR_THUMB,
  KEYPOINT_COLOR_INDEX, KEYPOINT_COLOR_INDEX, KEYPOINT_COLOR_INDEX, KEYPOINT_COLOR_INDEX,
  KEYPOINT_COLOR_MIDDLE, KEYPOINT_COLOR_MIDDLE, KEYPOINT_COLOR_MIDDLE, KEYPOINT_COLOR_MIDDLE,
  KEYPOINT_COLOR_RING, KEYPOINT_COLOR_RING, KEYPOINT_COLOR_RING, KEYPOINT_COLOR_RING,
  KEYPOINT_COLOR_PINKY, KEYPOINT_COLOR_PINKY, KEYPOINT_COLOR_PINKY, KEYPOINT_COLOR_PINKY,
  KEYPOINT_COLOR_WRIST, KEYPOINT_COLOR_WRIST, KEYPOINT_COLOR_WRIST,
  KEYPOINT_COLOR_CENTER,
  KEYPOINT_COLOR_INDEX,
  KEYPOINT_COLOR_MIDDLE,
  KEYPOINT_COLOR_RING,
  KEYPOINT_COLOR_PINKY
}};
}  // namespace

float* differential(float v1[], float v2[]){
  float diff[3] = {v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]};
  return diff;
}
float magnitude(float v[]){
  return sqrt(pow(v[0],2) + pow(v[1],2) + pow(v[2],2));
}
float distance(float v1[], float v2[]){
  return magnitude(differential(v1, v2));
}
float* normalize(float v[], float mult){
  float m = mult/magnitude(v);
  float v2[3];
  for(uint i=0; i<3; i++) v2[i] = v[i]*m;
  return v2;
}

class Particle {
public:
    float x[3];
    float v[3] = {0,0,0};

    Particle(){
      for(uint i=0; i<3; i++) x[i] = SPACE_SCALE*(rand()-.5)*2;
    }

    void boost(float *x2, float *x1){
      float* diff = differential(x2,x1);
      float sep = distance(x2, this->x);
      float* F = normalize(diff, exp(-pow(magnitude(differential(x2,this->x)),2)));

      for(uint i=0; i<3; i++) this->v[i] += F[i]*DT;
    }
    void update(){
      for(uint i=0; i<3; i++){
        this->x[i] += v[i];
        this->v[i] *= DRAG;
      }
    }
};

class HandData {
public:
    float x[2][28][3];

    MLHandTrackingStaticData hand_static_data_ = {};
    HandData(){
      MLHandTrackingStaticDataInit(&hand_static_data_);
    }

    const MLCoordinateFrameUID &GetHandFrameId(const MLHandTrackingHandType hand_type, const int idx) const {
      return hand_static_data_.hand_cfuids[hand_type].keypoint_cfuids[idx];
    }

    void copy(MLSnapshot *snapshot){
      for(uint hand_type=0; hand_type<2; hand_type++) {
        for (uint i = 0; i < 2; ++i) {
          MLTransform keypoint_transform;
          UNWRAP_MLRESULT(MLSnapshotGetTransform(snapshot, &GetHandFrameId(static_cast<MLHandTrackingHandType>(hand_type), static_cast<MLHandTrackingHandType>(i)),
                                                 &keypoint_transform));
          const glm::vec3 joint_position = ml::app_framework::to_glm(
                  keypoint_transform.position);
          for (uint k = 0; k < 3; k++) this->x[hand_type][i][k] = joint_position[k];
        }
      }
    }
};
HandData* data_prev = new HandData();

class HandTrackingApp : public ml::app_framework::Application {
public:
  HandTrackingApp(struct android_app *state)
      : ml::app_framework::Application(state, USE_GUI), hand_tracker_(ML_INVALID_HANDLE), draw_labels_(false), color_bones_(false) {
    MLHandTrackingStaticDataInit(&hand_static_data_);
  }

  void OnStart() override {
    MLHandTrackingSettings settings;
    MLHandTrackingSettingsInit(&settings);
    UNWRAP_MLRESULT(MLHandTrackingCreateEx(&settings, &hand_tracker_));
    UNWRAP_MLRESULT_FATAL(MLHandTrackingGetStaticData(hand_tracker_, &hand_static_data_));
    CreateMeshes();
    GetGui().Show();

    MLHandTrackingData data;
    MLHandTrackingDataInit(&data);
    MLSnapshot *snapshot = nullptr;
    MLPerceptionGetSnapshot(&snapshot);
    ASSERT_MLRESULT(MLHandTrackingGetData(hand_tracker_, &data));
    data_prev->copy(snapshot);
  }

  void OnResume() override {
    SetColorOverride(color_bones_);
  }

  void OnStop() override {
    UNWRAP_MLRESULT(MLHandTrackingDestroy(hand_tracker_));
    hand_tracker_ = ML_INVALID_HANDLE;
  }

  void OnUpdate(float) override {
    if (hand_tracker_ == ML_INVALID_HANDLE) {
      return;
    }

    MLHandTrackingData data;
    MLHandTrackingDataInit(&data);
    MLSnapshot *snapshot = nullptr;
    MLPerceptionGetSnapshot(&snapshot);

    ResetDrawFlags();

    ASSERT_MLRESULT(MLHandTrackingGetData(hand_tracker_, &data));
    UpdateHand(snapshot, data, MLHandTrackingHandType_Left);
    UpdateHand(snapshot, data, MLHandTrackingHandType_Right);

    UpdateParticles(snapshot);

    data_prev->copy(snapshot);

    MLPerceptionReleaseSnapshot(snapshot);

    UpdateFocusDistance();
    UpdateGui(data);
  }

private:
  using NodePtr = std::shared_ptr<ml::app_framework::Node>;
  using KeypointsNodeArray = std::array<NodePtr, MLHandTrackingStaticData_MaxKeyPoints>;
  using KeypointsVec3Array = std::array<glm::vec3, MLHandTrackingStaticData_MaxKeyPoints>;

  std::array<KeypointsNodeArray, MLHandTrackingHandType_Count> hand_labels_;
  std::array<KeypointsNodeArray, MLHandTrackingHandType_Count> hand_joints_;
  std::array<Particle*, N_PARTICLES> particles_;
  std::array<KeypointsVec3Array, MLHandTrackingHandType_Count> hand_keypoints_position_;
  std::array<HandSkeleton, MLHandTrackingHandType_Count> hand_bones_;
  MLHandle hand_tracker_ = ML_INVALID_HANDLE;
  MLHandTrackingStaticData hand_static_data_ = {};
  bool draw_labels_, color_bones_;

  // Updates focus distance so that MR recordings are better composited, if done
  void UpdateFocusDistance() {
    const float dist_l = glm::length(hand_joints_[MLHandTrackingHandType_Left][MLHandTrackingKeyPoint_Hand_Center]->GetLocalTranslation());
    const float dist_r = glm::length(hand_joints_[MLHandTrackingHandType_Right][MLHandTrackingKeyPoint_Hand_Center]->GetLocalTranslation());
    const float distance = (dist_l + dist_r) / 2.f;
    if (distance > 0.f) {
      auto frame_params = GetFrameParams();
      frame_params.focus_distance = distance;
      UpdateFrameParams(frame_params);
    }
  }

  void CreateMeshes() {
    for (uint8_t hand_idx = 0; hand_idx < MLHandTrackingHandType_Count; ++hand_idx) {
      const MLHandTrackingHandType hand_type = static_cast<MLHandTrackingHandType>(hand_idx);
      for (uint8_t i = 0; i < MLHandTrackingStaticData_MaxKeyPoints; ++i) {
        hand_labels_[hand_type][i] = ml::app_framework::CreatePresetNode(ml::app_framework::NodeType::Text);
        const std::string hand_label = GetHandLabel(hand_type)[0] + std::to_string(i);
        hand_labels_[hand_type][i]->GetComponent<ml::app_framework::TextComponent>()->SetText(hand_label.c_str());
        hand_labels_[hand_type][i]->SetLocalScale(KEYPOINT_LABEL_SCALE);
        GetRoot()->AddChild(hand_labels_[hand_type][i]);

        hand_joints_[hand_type][i] = ml::app_framework::CreatePresetNode(ml::app_framework::NodeType::Cube);
        auto mat = hand_joints_[hand_type][i]->GetComponent<ml::app_framework::RenderableComponent>()->GetMaterial();
        auto flat_mat = std::static_pointer_cast<ml::app_framework::FlatMaterial>(mat);
        flat_mat->SetOverrideVertexColor(color_bones_);
        flat_mat->SetColor(KEYPOINT_COLORS[i]);
        hand_joints_[hand_type][i]->SetLocalScale(KEYPOINT_CUBE_SCALE);
        GetRoot()->AddChild(hand_joints_[hand_type][i]);
      }
      hand_bones_[hand_type].Initialize(GetRoot());
    }

    // Create particles
    for(uint8_t i=0; i<N_PARTICLES; i++){
      particles_[i] = new Particle();
    }

    // Get initial hand state and save it
    // (used to calculate hand velocity later when hand state is updated)
    MLHandTrackingData data;
    MLHandTrackingDataInit(&data);
    MLSnapshot *snapshot = nullptr;
    MLPerceptionGetSnapshot(&snapshot);

    ResetDrawFlags();

    ASSERT_MLRESULT(MLHandTrackingGetData(hand_tracker_, &data));
    data_prev->copy(snapshot);
  }

  const MLCoordinateFrameUID &GetHandFrameId(const MLHandTrackingHandType hand_type, const int idx) const {
    return hand_static_data_.hand_cfuids[hand_type].keypoint_cfuids[idx];
  }

  void UpdateHand(MLSnapshot *snapshot, const MLHandTrackingData &data, MLHandTrackingHandType hand_type) {
    const MLHandTrackingHandState &hand_state = data.hand_state[hand_type];
    bool hand_valid = hand_state.hand_confidence > 0;
    if (hand_valid){
      for (uint8_t i = 0; i < MLHandTrackingStaticData_MaxKeyPoints; ++i) {
        if (hand_state.keypoints_mask[i]) {
          MLTransform keypoint_transform;
          UNWRAP_MLRESULT(MLSnapshotGetTransform(snapshot, &GetHandFrameId(hand_type, i), &keypoint_transform));
          const glm::vec3 joint_position = ml::app_framework::to_glm(keypoint_transform.position);

          auto joint = hand_joints_[hand_type][i];
          joint->SetWorldTranslation(joint_position);
          joint->SetWorldRotation(ml::app_framework::to_glm(keypoint_transform.rotation));
          joint->GetComponent<ml::app_framework::RenderableComponent>()->SetVisible(true);
          hand_keypoints_position_[hand_type][i] = joint_position;

          for(uint j=0; j<3; j++) data_prev->x[hand_type][i][j] = hand_keypoints_position_[hand_type][i][j];

          auto label = hand_labels_[hand_type][i];
          label->SetWorldTranslation(joint_position);
          label->GetComponent<ml::app_framework::RenderableComponent>()->SetVisible(draw_labels_);
        }
      }
    }
    hand_bones_[hand_type].Update(hand_keypoints_position_[hand_type], hand_valid);
  }

  void UpdateParticles(MLSnapshot *snapshot){
    for(uint hand_type=0; hand_type<2; hand_type++) {
      for (uint8_t i = 0; i < MLHandTrackingStaticData_MaxKeyPoints; ++i) {
        MLTransform keypoint_transform;
        UNWRAP_MLRESULT(MLSnapshotGetTransform(snapshot, &GetHandFrameId(static_cast<MLHandTrackingHandType>(hand_type), static_cast<MLHandTrackingHandType>(i)),
                                               &keypoint_transform));
        const glm::vec3 joint_position = ml::app_framework::to_glm(keypoint_transform.position);

        float cur[3];
        for(uint el=0; el<3; el++){
          cur[el] = joint_position[el];
        }
        for (uint i = 0; i < N_PARTICLES; i++) {
          particles_[i]->boost(cur, data_prev->x[hand_type][i]);
        }
      }
    }

    for(uint8_t i=0; i<N_PARTICLES; i++){
      particles_[i]->update();
    }
  }

  void ResetDrawFlags() {
    for (uint8_t hand_idx = 0; hand_idx < MLHandTrackingHandType_Count; ++hand_idx) {
      const MLHandTrackingHandType hand_type = static_cast<MLHandTrackingHandType>(hand_idx);
      for (uint8_t i = 0; i < MLHandTrackingStaticData_MaxKeyPoints; ++i) {
        hand_joints_[hand_type][i]->GetComponent<ml::app_framework::RenderableComponent>()->SetVisible(false);
        hand_labels_[hand_type][i]->GetComponent<ml::app_framework::RenderableComponent>()->SetVisible(false);
      }
    }
  }

  static const char *GetHandLabel(const MLHandTrackingHandType &hand_type) {
    switch (hand_type) {
      case MLHandTrackingHandType_Left: return "Left";
      case MLHandTrackingHandType_Right: return "Right";
      default: return "Invalid hand type";
    }
  }

  static std::pair<const char *, const char *> GetKeypointLabel(const MLHandTrackingKeyPoint &keypoint) {
    switch (keypoint) {
      case MLHandTrackingKeyPoint_Thumb_Tip: return {"Thumb", "Tip"};
      case MLHandTrackingKeyPoint_Thumb_IP: return {"Thumb", "IP"};
      case MLHandTrackingKeyPoint_Thumb_MCP: return {"Thumb", "MCP"};
      case MLHandTrackingKeyPoint_Thumb_CMC: return {"Thumb", "CMC"};
      case MLHandTrackingKeyPoint_Index_Tip: return {"Index", "Tip"};
      case MLHandTrackingKeyPoint_Index_DIP: return {"Index", "DIP"};
      case MLHandTrackingKeyPoint_Index_PIP: return {"Index", "PIP"};
      case MLHandTrackingKeyPoint_Index_MCP: return {"Index", "MCP"};
      case MLHandTrackingKeyPoint_Middle_Tip: return {"Middle", "Tip"};
      case MLHandTrackingKeyPoint_Middle_DIP: return {"Middle", "DIP"};
      case MLHandTrackingKeyPoint_Middle_PIP: return {"Middle", "PIP"};
      case MLHandTrackingKeyPoint_Middle_MCP: return {"Middle", "MCP"};
      case MLHandTrackingKeyPoint_Ring_Tip: return {"Ring", "Tip"};
      case MLHandTrackingKeyPoint_Ring_DIP: return {"Ring", "DIP"};
      case MLHandTrackingKeyPoint_Ring_PIP: return {"Ring", "PIP"};
      case MLHandTrackingKeyPoint_Ring_MCP: return {"Ring", "MCP"};
      case MLHandTrackingKeyPoint_Pinky_Tip: return {"Pinky", "Tip"};
      case MLHandTrackingKeyPoint_Pinky_DIP: return {"Pinky", "DIP"};
      case MLHandTrackingKeyPoint_Pinky_PIP: return {"Pinky", "PIP"};
      case MLHandTrackingKeyPoint_Pinky_MCP: return {"Pinky", "MCP"};
      case MLHandTrackingKeyPoint_Wrist_Center: return {"Wrist", "Center"};
      case MLHandTrackingKeyPoint_Wrist_Ulnar: return {"Wrist", "Ulnar"};
      case MLHandTrackingKeyPoint_Wrist_Radial: return {"Wrist", "Radial"};
      case MLHandTrackingKeyPoint_Hand_Center: return {"Hand", "Center"};
      case MLHandTrackingKeyPoint_Index_Meta: return {"Index", "Meta"};
      case MLHandTrackingKeyPoint_Middle_Meta: return {"Middle", "Meta"};
      case MLHandTrackingKeyPoint_Ring_Meta: return {"Ring", "Meta"};
      case MLHandTrackingKeyPoint_Pinky_Meta: return {"Pinky", "Meta"};
      default: return {"", "Invalid keypoint"};
    }
  }

  void SetColorOverride(bool draw_colors) {
    for (uint8_t hand_idx = 0; hand_idx < MLHandTrackingHandType_Count; ++hand_idx) {
      const MLHandTrackingHandType hand_type = static_cast<MLHandTrackingHandType>(hand_idx);
      for (uint8_t i = 0; i < MLHandTrackingStaticData_MaxKeyPoints; ++i) {
        auto mat = hand_joints_[hand_type][i]->GetComponent<ml::app_framework::RenderableComponent>()->GetMaterial();
        auto flat_mat = std::static_pointer_cast<ml::app_framework::FlatMaterial>(mat);
        flat_mat->SetOverrideVertexColor(draw_colors);
      }
      hand_bones_[hand_type].SetColorOverride(draw_colors);
    }
  }

  void UpdateGui(const MLHandTrackingData &data) {
    auto &gui = GetGui();
    gui.BeginUpdate();
    if (gui.BeginDialog("HandTracking")) {
      ImGui::Checkbox("Show keypoint labels", &draw_labels_);
      if (ImGui::Checkbox("Color fingers", &color_bones_)) {
        SetColorOverride(color_bones_);
      }
      if (ImGui::CollapsingHeader("MLHandTrackingData", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (uint8_t hand_idx = 0; hand_idx < MLHandTrackingHandType_Count; ++hand_idx) {
          const MLHandTrackingHandType hand_type = static_cast<MLHandTrackingHandType>(hand_idx);
          const auto &state = data.hand_state[hand_type];
          const char *hand_label = GetHandLabel(hand_type);
          if (ImGui::BeginChild(hand_label, ImVec2(ImGui::GetWindowContentRegionWidth() * 0.5f, 0.f)) &&
              ImGui::CollapsingHeader(hand_label, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Value("Hand confidence", state.hand_confidence, "%.2f");
            if (ImGui::TreeNodeEx("Keypoints mask")) {
              for (int point = 0; point < MLHandTrackingStaticData_MaxKeyPoints; ++point) {
                const auto label = GetKeypointLabel(static_cast<MLHandTrackingKeyPoint>(point));
                ImGui::Text("[%2d] %-6s %-6s: %s", point, label.first, label.second,
                            state.keypoints_mask[point] ? "true" : "false");
              }
              ImGui::TreePop();
            }
          }
          ImGui::EndChild();
          ImGui::SameLine();
        }
      }
    }
    gui.EndDialog();
    gui.EndUpdate();
  }
};

void android_main(struct android_app *state) {
  HandTrackingApp app(state);
  app.RunApp();
}
