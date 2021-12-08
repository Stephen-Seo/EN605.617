#ifndef IGPUP_MODULE_13_EVENTS_COMMAND_QUEUES_DEPENDENCIES_H_
#define IGPUP_MODULE_13_EVENTS_COMMAND_QUEUES_DEPENDENCIES_H_

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern const std::unordered_set<unsigned int> kEmptySet;

typedef std::unordered_map<unsigned int, std::unordered_set<unsigned int>> DMap;

class ReverseDependencies {
 public:
  // allow copy
  ReverseDependencies(const ReverseDependencies &other) = default;
  ReverseDependencies &operator=(const ReverseDependencies &other) = default;

  // allow move
  ReverseDependencies(ReverseDependencies &&other) = default;
  ReverseDependencies &operator=(ReverseDependencies &&other) = default;

  std::unordered_set<unsigned int> GetDependencies() const;
  const std::unordered_set<unsigned int> &GetReverseDependents(
      unsigned int to) const;

  bool IsEmpty() const;

  std::vector<std::vector<unsigned int>> GetDependenciesOrdered() const;

 private:
  friend class Dependencies;
  ReverseDependencies();
  ReverseDependencies(DMap reverse_deps, DMap deps);

  DMap reverse_deps_;
  DMap deps_;
};

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

  std::unordered_set<unsigned int> GetDependents() const;
  const std::unordered_set<unsigned int> &GetDependencies(
      unsigned int from) const;

  bool IsEmpty() const;

  std::unique_ptr<unsigned int> HasCycle() const;

  ReverseDependencies GenerateReverseDependencies() const;

 private:
  DMap deps_;

  bool CycleFromValue(unsigned int value) const;
};

#endif
