// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "controllers/bundle_adjustment.h"

#include <ceres/ceres.h>

#include "base/gps.h"
#include "optim/bundle_adjustment.h"
#include "util/misc.h"

namespace colmap {
namespace {

// Callback functor called after each bundle adjustment iteration.
class BundleAdjustmentIterationCallback : public ceres::IterationCallback {
 public:
  explicit BundleAdjustmentIterationCallback(Thread* thread)
      : thread_(thread) {}

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    CHECK_NOTNULL(thread_);
    thread_->BlockIfPaused();
    if (thread_->IsStopped()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    } else {
      return ceres::SOLVER_CONTINUE;
    }
  }

 private:
  Thread* thread_;
};

}  // namespace

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options, Reconstruction* reconstruction)
    : options_(options), reconstruction_(reconstruction) {}

bool BundleAdjustmentController::SetUpPriorMotions() {

  PrintHeading1("Loading database");

  std::string database_path = options_.bundle_adjustment->database_path;
  Database database(database_path);
  Timer timer;
  timer.Start();
  database_cache_.Load_Images(database,reconstruction_->RegImageIds());
  std::cout << std::endl;
  timer.PrintMinutes();

  std::cout << std::endl;

  if (database_cache_.NumImages() == 0) {
    std::cout << "WARNING: No images found in the database."
              << std::endl
              << std::endl;
    return false;
  }

  GPSTransform gps_transform(GPSTransform::WGS84);
  size_t nb_motion_prior = 0;

  std::cout << "\nSetting Up Prior Motions!\n";

  // GPS to ENU conversion
  if (options_.bundle_adjustment->prior_is_gps && options_.bundle_adjustment->use_enu_coords) {
    // Get image ids to get ordered GPS prior coords.
    std::set<image_t> image_prior_ids;

    for (const auto& image : database_cache_.Images()) {
      if (image.second.HasTvecPrior()) {
        image_prior_ids.insert(image.first);
        ++nb_motion_prior;
      }
    }

    // Get ordered GPS prior coords.
    std::vector<Eigen::Vector3d> vgps_priors;
    vgps_priors.reserve(database_cache_.NumImages());

    for (const auto image_id : image_prior_ids) {
      vgps_priors.push_back(database_cache_.Image(image_id).TvecPrior());
    }

    // Convert to GPS to ENU coords.
    std::vector<Eigen::Vector3d> vtvec_priors =
        gps_transform.EllToENU(vgps_priors);

    // Set back the prior coords.
    size_t k = 0;
    for (const auto image_id : image_prior_ids) {
      database_cache_.Image(image_id).SetTvecPrior(vtvec_priors[k]);
      k++;
    }
  } else {
    for (auto& image : database_cache_.Images()) {
      if (image.second.HasTvecPrior()) {
        if (options_.bundle_adjustment->prior_is_gps) {
          // GPS to ECEF conversion
          database_cache_.Image(image.first)
              .SetTvecPrior(
                  gps_transform.EllToXYZ({image.second.TvecPrior()})[0]);
        }
        ++nb_motion_prior;
      }
    }
  }

  if (nb_motion_prior < database_cache_.NumImages()) {
    std::cout << "WARNING! Not all images have a motion prior!\n";

    if (nb_motion_prior < 3) {
      std::cout << StringPrintf(
          "Only %d motion priors! Cannot optimize with use_motion_prior "
          "checked...",
          nb_motion_prior);
      return false;
    }
  }

  for (auto &&image_id : reconstruction_->RegImageIds()) {
    reconstruction_->Image(image_id).SetTvecPrior(database_cache_.Image(image_id).TvecPrior());
  }

  return true;
}

void BundleAdjustmentController::Run() {
  CHECK_NOTNULL(reconstruction_);

  PrintHeading1("Global bundle adjustment");

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  if (reg_image_ids.size() < 2) {
    std::cout << "ERROR: Need at least two views." << std::endl;
    return;
  }

  if (options_.bundle_adjustment->use_prior_motion) {
    std::cout << "\nSETTING UP PRIOR MOTION!";
    if (!SetUpPriorMotions()) {
      return;
    }
  }

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
  ba_options.solver_options.minimizer_progress_to_stdout = true;

  BundleAdjustmentIterationCallback iteration_callback(this);
  ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  if (!ba_options.use_prior_motion) {
    ba_config.SetConstantPose(reg_image_ids[0]);
    ba_config.SetConstantTvec(reg_image_ids[1], {0});
  }

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  bundle_adjuster.Solve(reconstruction_);

  GetTimer().PrintMinutes();
}

}  // namespace colmap
