#ifndef IGPUP_MODULE_13_EVENTS_COMMAND_QUEUES_DEPENDENCIES_H_
#define IGPUP_MODULE_13_EVENTS_COMMAND_QUEUES_DEPENDENCIES_H_

#include <memory>
#include <unordered_map>
#include <unordered_set>

class Dependencies {
 public:
  Dependencies();

  // allow copy
  Dependencies(const Dependencies &other) = default;
  Dependencies &operator=(const Dependencies &other) = default;

  // allow move
  Dependencies(Dependencies &&other) = default;
  Dependencies &operator=(Dependencies &&other) = default;

  /// Returns true if specified dependency was added and did not already exist
  bool AddDependency(unsigned int from, unsigned int to);
  /// Returns true if specified dependency was found and removed
  bool RemoveDepenency(unsigned int from, unsigned int to);

  const std::unordered_set<unsigned int> &GetDependencies(
      unsigned int from) const;

  bool IsEmpty() const;

  std::unique_ptr<unsigned int> HasCycle() const;

 private:
  std::unordered_map<unsigned int, std::unordered_set<unsigned int>> deps_;
  static const std::unordered_set<unsigned int> empty_set_;

  bool CycleFromValue(unsigned int value) const;
};

#endif
