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

#include <ml_head_tracking.h>
#include <ml_hand_tracking.h>
#include <ml_perception.h>

#include <imgui.h>

namespace {
    constexpr glm::vec3 PARTICLE_SCALE = glm::vec3(.003f, .003f, .003f);
    constexpr glm::vec4 PARTICLE_COLOR = glm::vec4(.7f, .7f, .0f,1.f);

// Define some physical constants that determine strength of particle-particle, particle-hand, and cloud confinement forces (as well as some other things)
    const float FORCE_HAND = .5;
    const float FORCE_INTER = .00001;
    const float FORCE_CONFINE = .2;
    const float HAND_SIZE = .1;
    const int N_PARTICLES = 200;
    const float_t SPACE_SCALE = .2;
    const float DT = 0.1;
    const float THRESH_CONFIDENCE = .2;

// Generates random floats between -lim and lim
    float randf(float lim) {
        return lim * 2.0 * (rand() / (RAND_MAX + 1.0) - 0.5);
    }

// Function for displaying checkpoints (nuclear option for troubleshooting)
    void cp(float i) {
        ALOG(ANDROID_LOG_INFO, "MyTag", "cp:%3.3f", i);
    }

    glm::vec3 hand_data_prev;

// This is where the particle cloud will be centered, initially
    glm::vec3 origin = glm::vec3(0.0f, 0.0f, -.5f);

// Particle class that holds and updates particle information
    class Particle {
    public:
        using NodePtr = std::shared_ptr<ml::app_framework::Node>;
        glm::vec3 x;
        glm::vec3 v;
        glm::vec3 F;
        NodePtr obj = ml::app_framework::CreatePresetNode(ml::app_framework::NodeType::Cube);

        Particle() {
            this->reset();
            this->zeroF();
        }

        void reset() {
            this->x = origin;
            this->x.x += randf(SPACE_SCALE);
            this->x.y += randf(SPACE_SCALE);
            this->x.z += randf(SPACE_SCALE);

            this->v = glm::vec3(0);
            this->zeroF();
        }

        void zeroF() {
            this->F = glm::vec3(0);
        }

        void boost(glm::vec3 force) {
            this->F += force;
        }

        float boost_hand(glm::vec3 hand_pos, glm::vec3 hand_vel) {
            // Make sure that hand_vel is not (0,0,0). Otherwise, the force is undefined
            // The hand force takes the form of a Gaussian field pointing in the direction of hand velocity
            float mult = FORCE_HAND * exp(-pow(glm::distance(hand_pos, this->x) / HAND_SIZE, 2));
            glm::vec3 Fhand = glm::normalize(hand_vel) * mult;
            this->boost(Fhand);
            return glm::length(Fhand);
        }

        void update(float drag) {
            /*auto el = this;
            ALOG(ANDROID_LOG_INFO, "MyTag", "kpt.position = %3.3f, %3.3f, %3.3f kpd.velocity = %3.3f, %3.3f, %3.3f",
                 el->x.x,
                 el->x.y,
                 el->x.z,
                 el->v.x,
                 el->v.y,
                 el->v.z);*/

            // Basic Newtonian stuff
            this->v += this->F * DT;
            this->x += this->v * DT;
            this->v *= drag;

            // Update rendered object
            this->reposition();

            // Set force to zero in preparation for next timestep where forces will be recalculated
            this->zeroF();
        }

        void reposition() {
            // Update the position of particle in the node tree
            this->obj->SetWorldTranslation(this->x);
        }
    };

    // The Swarm class holds all particles and gets angry if you try to grab it with 2 hands
    class Swarm {
    public:
        using Node = ml::app_framework::Node;
        std::array<Particle *, N_PARTICLES> particles;
        Particle *origin;
        float anger = 0.199f;
        float angerOff = 0.2f;
        float drag_particle_baseline = 0.98;
        float drag_particle_angry = 1;
        float drag_swarm_baseline = 0.992;
        float drag_swarm_angry = .994;
        float anger_decay = .999;
        float F_anger_thresh = 0.49; // Hand force above which the swarm is angered
        float chaos_swarm = 0.06;
        float chaos_particle = 0.1;

        Swarm(const std::shared_ptr<Node> root) {
            // Create particles
            for (uint8_t i = 0; i < N_PARTICLES; i++) {
                // The Particle class holds and updates particle data
                this->particles[i] = new Particle();
                this->create_mesh(root, this->particles[i]);
            }
            // Let the entire particle cloud act as a particle that can move around
            this->origin = new Particle();
            this->origin->x = ::origin; // x is initially randomized for a Particle, but we want it to be placed at the origin for the cloud
        }

        void create_mesh(std::shared_ptr<Node> root, Particle* p) {
            // The following will create the particle on the node tree
            // Each particle is a cube with some default texture
            // Ideally it would be rendered using a low-level shader, enabling me to add a lot more particles
            p->obj->GetComponent<ml::app_framework::RenderableComponent>()->SetVisible(true);
            auto mat = p->obj->GetComponent<ml::app_framework::RenderableComponent>()->GetMaterial();
            auto flat_mat = std::static_pointer_cast<ml::app_framework::FlatMaterial>(mat);
            flat_mat->SetOverrideVertexColor(true);
            flat_mat->SetColor(PARTICLE_COLOR);
            p->obj->SetLocalScale(PARTICLE_SCALE);
            root->AddChild(p->obj);
        }

        void force_hand(glm::vec3 pos, glm::vec3 vel){
            // Add hand force to each particle
            // Add the same force to the entire system (cloud_origin)
            //float Fmax = 0;
            if (glm::length(vel) > 0) {
                for (uint i = 0; i < N_PARTICLES; i++) {
                    float Fcur = this->particles[i]->boost_hand(pos, vel);
                    if(Fcur > this->F_anger_thresh) this->anger = 1.0;
                    //if(Fcur > Fmax) Fmax = Fcur;
                    this->origin->boost(this->particles[i]->F / float(N_PARTICLES * 2));
                }
            }
            //ALOG(ANDROID_LOG_INFO, "MyTag", "anger = %3.2f", this->anger);
        }

        void force_head(glm::vec3 pos){
            /*auto el = pos;
            auto el2 = this->origin->x;
            ALOG(ANDROID_LOG_INFO, "MyTag", "anger=%3.3f, v1 = %3.3f, %3.3f, %3.3f v2 = %3.3f, %3.3f, %3.3f",
                 float(pow(this->anger,2)),
                 el.x,
                 el.y,
                 el.z,
                 el2.x,
                 el2.y,
                 el2.z);*/
            // If the angering just happened, then the swarm lunges away
            // After that, it comes at the user's head.
            // Note that the force magnitude is not dependent on distance to user
            if(this->anger == 1.0)
                this->origin->boost( -.4f*glm::normalize(pos - this->origin->x));
            else if(this->anger > this->angerOff)
                this->origin->boost(glm::normalize(pos - this->origin->x) * 0.02f);
        }

        void force_other(){
            // Calculate inter-particle repulsive force
            for (uint8_t i = 1; i < N_PARTICLES; i++) {
                for (uint8_t j = 0; j < i; j++) {
                    glm::vec3 rad = this->particles[i]->x - this->particles[j]->x;
                    float denom = pow(glm::length(rad), 3);
                    glm::vec3 Finter = rad * FORCE_INTER / denom;
                    this->particles[i]->boost(Finter);
                    this->particles[j]->boost(-Finter);
                }
            }

            // Calculate cloud confinement force
            // Update each particle's velocities and positions
            for (uint8_t i = 0; i < N_PARTICLES; i++) {
                this->particles[i]->boost(FORCE_CONFINE * (this->origin->x - this->particles[i]->x));
                if(this->anger > angerOff)
                    this->particles[i]->boost(this->chaos_particle * glm::vec3(randf(1.f), randf(1.f), randf(1.f)));
            }

            // Make the origin move chaotically in proportion to anger
            if(this->anger > angerOff)
                this->origin->boost(this->chaos_swarm * glm::vec3(randf(1.f), randf(1.f), randf(1.f)));
        }

        // Functions for calculating drag coefficients
        float drag_particle(){
            return this->drag_particle_angry*this->anger + this->drag_particle_baseline*(1-this->anger);
        }
        float drag_swarm(){
            return this->drag_swarm_angry*this->anger + this->drag_swarm_baseline*(1-this->anger);
        }

        // Time update function
        void update(){
            // Update particles
            for (uint8_t i = 0; i < N_PARTICLES; i++) {
                // The Particle class holds and updates particle data
                this->particles[i]->update(this->drag_particle());
            }

            this->origin->update(this->drag_swarm());

            this->anger *= this->anger_decay;
        }
    };

    class HandTrackingApp : public ml::app_framework::Application {
    public:
        HandTrackingApp(struct android_app *state)
                : ml::app_framework::Application(state), hand_tracker_(ML_INVALID_HANDLE),
                  draw_labels_(false), color_bones_(false) {
            MLHandTrackingStaticDataInit(&hand_static_data_);
        }

        void OnStart() override {
            MLHandTrackingSettings settings;
            MLHandTrackingSettingsInit(&settings);
            UNWRAP_MLRESULT(MLHandTrackingCreateEx(&settings, &hand_tracker_));
            UNWRAP_MLRESULT_FATAL(MLHandTrackingGetStaticData(hand_tracker_, &hand_static_data_));

            hand_data_prev = glm::vec3(0);

            MLHeadTrackingStateExInit(&prev_head_state_);
            UNWRAP_MLRESULT(MLHeadTrackingCreate(&head_tracker_));
            UNWRAP_MLRESULT(MLHeadTrackingGetStaticData(head_tracker_, &head_static_data_));

            swarm = new Swarm(GetRoot());
        }

        void OnResume() override {
            //SetColorOverride(color_bones_);
        }

        void OnStop() override {
            UNWRAP_MLRESULT(MLHandTrackingDestroy(hand_tracker_));
            hand_tracker_ = ML_INVALID_HANDLE;

            UNWRAP_MLRESULT(MLHeadTrackingDestroy(head_tracker_));
            head_tracker_ = ML_INVALID_HANDLE;
        }

        void OnUpdate(float) override {
            if (hand_tracker_ == ML_INVALID_HANDLE || head_tracker_ == ML_INVALID_HANDLE) {
                return;
            }

            // MLHandTrackingData object will tell whether data was collected
            MLHandTrackingData data;
            MLHandTrackingDataInit(&data);
            // MLSnapshot object is a timestamp that we use to extract cached data later
            MLSnapshot *snapshot = nullptr;
            MLPerceptionGetSnapshot(&snapshot);

            ASSERT_MLRESULT(MLHandTrackingGetData(hand_tracker_, &data));

            MLHeadTrackingStateEx cur_state;
            MLHeadTrackingStateExInit(&cur_state);
            UNWRAP_MLRESULT(MLHeadTrackingGetStateEx(head_tracker_, &cur_state));

            if (!initial_state_logged_ || prev_head_state_.status != cur_state.status ||
                prev_head_state_.error != cur_state.error ||
                std::fabs(prev_head_state_.confidence - cur_state.confidence) > 0.3f) {
                    initial_state_logged_ = true;
            }
            prev_head_state_ = cur_state;

            // I copied/modified UpdateHand to collect similar information and use it to update particle velocities
            UpdateParticles(snapshot, data);

            MLPerceptionReleaseSnapshot(snapshot);
        }

    private:
        using NodePtr = std::shared_ptr<ml::app_framework::Node>;
        using KeypointsNodeArray = std::array<NodePtr, MLHandTrackingStaticData_MaxKeyPoints>;
        using KeypointsVec3Array = std::array<glm::vec3, MLHandTrackingStaticData_MaxKeyPoints>;

        std::array<KeypointsNodeArray, MLHandTrackingHandType_Count> hand_joints_;
        std::array<KeypointsVec3Array, MLHandTrackingHandType_Count> hand_keypoints_position_;
        std::array<HandSkeleton, MLHandTrackingHandType_Count> hand_bones_;
        MLHandle hand_tracker_ = ML_INVALID_HANDLE;
        MLHandTrackingStaticData hand_static_data_ = {};
        bool draw_labels_, color_bones_;
        std::vector<uint8_t> active_joints = {MLHandTrackingKeyPoint_Hand_Center};

        Swarm* swarm;

        MLHandle head_tracker_;
        MLHeadTrackingStaticData head_static_data_;
        bool initial_state_logged_ = false;
        MLHeadTrackingStateEx prev_head_state_;

        const MLCoordinateFrameUID &
        GetHandFrameId(const MLHandTrackingHandType hand_type, const int idx) const {
            return hand_static_data_.hand_cfuids[hand_type].keypoint_cfuids[idx];
        }

        void UpdateParticles(MLSnapshot *snapshot, const MLHandTrackingData &data) {
            MLTransform head_transform = {};
            UNWRAP_MLRESULT(MLSnapshotGetTransform(snapshot, &head_static_data_.coord_frame_head, &head_transform));
            UNWRAP_MLRESULT(MLPerceptionReleaseSnapshot(snapshot));
            const glm::vec3 head_pos = ml::app_framework::to_glm(head_transform.position);

            // For each hand...
            for (uint8_t hand_type = 0; hand_type < 2; hand_type++) {
                // ...get hand state
                const MLHandTrackingHandState &hand_state = data.hand_state[hand_type];
                // Check that hand data was collected
                if (hand_state.hand_confidence > THRESH_CONFIDENCE) {
                    // For each hand joint... (actually I'm only going to do this for one joint to speed things up)
                    for (uint8_t i = 0; i < active_joints.size(); i++) {
                        // ...check that hand joint data was collected
                        if (hand_state.keypoints_mask[active_joints[i]]) {
                            // Get hand data for a particular joint (Hand_Center) given the current snapshot (snapshot is just a timestamp)
                            MLTransform keypoint_transform;
                            MLSnapshotGetTransform(snapshot, &GetHandFrameId(
                                                           static_cast<MLHandTrackingHandType>(hand_type),
                                                           active_joints[i]),
                                                   &keypoint_transform);

                            // Convert to vec3 and calculate velocity based on previous hand position
                            const glm::vec3 pos = ml::app_framework::to_glm(keypoint_transform.position);
                            const glm::vec3 vel = pos - hand_data_prev;

                            swarm->force_hand(pos, vel);

                            // Save current position as previous position
                            hand_data_prev = pos;
                        }
                    }
                }
            }

            swarm->force_head(head_pos);

            swarm->force_other();
            swarm->update();
        }
    };
}

void android_main(struct android_app *state) {
  HandTrackingApp app(state);
  app.RunApp();
}