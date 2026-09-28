#include "face_detector.h"

#include <Arduino.h>
#include <list>
#include <vector>

#include "human_face_detect_mnp01.hpp"
#include "human_face_detect_msr01.hpp"

struct FaceDetector::Impl {
  HumanFaceDetectMSR01 proposal{0.08F, 0.60F, 40, 0.20F};
  HumanFaceDetectMNP01 refinement{0.40F, 0.45F, 20};
  size_t candidate_count = 0;
};

FaceDetector::FaceDetector() : impl_(new Impl()) {}

FaceDetector::~FaceDetector() {
  delete impl_;
}

bool FaceDetector::ready() const {
  return impl_ != nullptr;
}

size_t FaceDetector::last_candidate_count() const {
  return impl_ ? impl_->candidate_count : 0;
}

size_t FaceDetector::infer(const uint8_t *rgb565, int width, int height,
                           FaceBox *boxes, size_t capacity) {
  if (!ready() || boxes == nullptr || capacity == 0) return 0;

  auto &candidates = impl_->proposal.infer(
      reinterpret_cast<uint16_t *>(const_cast<uint8_t *>(rgb565)),
      {height, width, 3});
  impl_->candidate_count = candidates.size();
  auto &results = impl_->refinement.infer(
      reinterpret_cast<uint16_t *>(const_cast<uint8_t *>(rgb565)),
      {height, width, 3}, candidates);

  size_t count = 0;
  for (const auto &result : results) {
    if (count >= capacity) break;
    const int left = constrain(result.box[0], 0, width - 1);
    const int top = constrain(result.box[1], 0, height - 1);
    const int right = constrain(result.box[2], left, width - 1);
    const int bottom = constrain(result.box[3], top, height - 1);
    boxes[count++] = {
        .x = left,
        .y = top,
        .width = right - left + 1,
        .height = bottom - top + 1,
        .confidence = result.score,
    };
  }
  return count;
}
