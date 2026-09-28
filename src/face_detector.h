#pragma once

#include <stddef.h>
#include <stdint.h>

struct FaceBox {
  int x;
  int y;
  int width;
  int height;
  float confidence;
};

class FaceDetector {
 public:
  FaceDetector();
  ~FaceDetector();

  bool ready() const;
  size_t infer(const uint8_t *rgb565, int width, int height, FaceBox *boxes, size_t capacity);
  size_t last_candidate_count() const;

 private:
    struct Impl;
    Impl *impl_;
};
