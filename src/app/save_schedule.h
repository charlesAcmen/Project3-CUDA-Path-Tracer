#pragma once

#include <cstddef>
#include <utility>
#include <vector>

// Application-side state for one configured series of checkpoint images.
// A pass is one uninterrupted accumulation period; camera or integrator
// changes begin a new pass and replay the configured checkpoints.
class SaveSchedule
{
public:
    explicit SaveSchedule(std::vector<int> checkpoints = {})
        : checkpoints_(std::move(checkpoints))
    {
    }

    bool consumeCheckpointAt(int iteration)
    {
        if (nextCheckpoint_ >= checkpoints_.size()
            || iteration != checkpoints_[nextCheckpoint_])
        {
            return false;
        }

        ++nextCheckpoint_;
        return true;
    }

    // Completion is itself an automatic save. A checkpoint at the final
    // iteration is consumed but still produces one, not two, save requests.
    bool shouldSaveAt(int iteration, unsigned int finalIteration)
    {
        if (iteration == lastAutomaticSaveIteration_)
        {
            return false;
        }

        const bool shouldSave = consumeCheckpointAt(iteration)
            || (iteration >= 0 && static_cast<unsigned int>(iteration) == finalIteration);
        if (shouldSave)
        {
            lastAutomaticSaveIteration_ = iteration;
        }
        return shouldSave;
    }

    void beginNextPass()
    {
        ++pass_;
        nextCheckpoint_ = 0;
        lastAutomaticSaveIteration_ = -1;
    }

    unsigned int pass() const { return pass_; }

private:
    std::vector<int> checkpoints_;
    std::size_t      nextCheckpoint_ = 0;
    unsigned int     pass_           = 1;
    int              lastAutomaticSaveIteration_ = -1;
};
