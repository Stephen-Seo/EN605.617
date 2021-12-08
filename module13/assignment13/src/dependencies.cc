#include "dependencies.h"

#include <iostream>
#include <queue>

const std::unordered_set<unsigned int> kEmptySet = {};

ReverseDependencies::ReverseDependencies() : reverse_deps_() {}

ReverseDependencies::ReverseDependencies(DMap reverse_deps)
    : reverse_deps_(reverse_deps) {}

std::unordered_set<unsigned int> ReverseDependencies::GetDependencies() const {
  std::unordered_set<unsigned int> dependencies;

  for (const auto &pair : reverse_deps_) {
    dependencies.insert(pair.first);
  }

  return dependencies;
}

const std::unordered_set<unsigned int>
    &ReverseDependencies::GetReverseDependents(unsigned int to) const {
  auto iter = reverse_deps_.find(to);
  if (iter != reverse_deps_.end()) {
    return iter->second;
  } else {
    return kEmptySet;
  }
}

bool ReverseDependencies::IsEmpty() const { return reverse_deps_.empty(); }

Dependencies::Dependencies() : deps_() {}

bool Dependencies::AddDependency(unsigned int from, unsigned int to) {
  auto iter = deps_.find(from);
  if (iter == deps_.end()) {
    return deps_.insert({from, {to}}).second;
  } else if (iter->second.find(to) != iter->second.end()) {
    return false;
  } else {
    return iter->second.insert(to).second;
  }
}

bool Dependencies::RemoveDepenency(unsigned int from, unsigned int to) {
  auto iter = deps_.find(from);
  if (iter == deps_.end()) {
    return false;
  } else if (iter->second.find(to) == iter->second.end()) {
    return false;
  } else {
    bool removed = iter->second.erase(to) == 1;
    if (iter->second.empty()) {
      unsigned int prev_size = deps_.size();
      deps_.erase(iter);
      return prev_size > deps_.size();
    } else {
      return removed;
    }
  }
}

std::unordered_set<unsigned int> Dependencies::GetDependents() const {
  std::unordered_set<unsigned int> dependents;

  for (auto &pair : deps_) {
    dependents.insert(pair.first);
  }

  return dependents;
}

const std::unordered_set<unsigned int> &Dependencies::GetDependencies(
    unsigned int from) const {
  auto iter = deps_.find(from);
  if (iter == deps_.end()) {
    return kEmptySet;
  } else {
    return iter->second;
  }
}

bool Dependencies::IsEmpty() const { return deps_.empty(); }

std::unique_ptr<unsigned int> Dependencies::HasCycle() const {
  for (const auto &deps_pair : deps_) {
    if (CycleFromValue(deps_pair.first)) {
      return std::unique_ptr<unsigned int>(new unsigned int(deps_pair.first));
    }
  }

  return {};
}

ReverseDependencies Dependencies::GenerateReverseDependencies() const {
  Dependencies fauxDependencies{};

  for (const auto &pair : deps_) {
    for (unsigned int to : pair.second) {
      if (!fauxDependencies.AddDependency(to, pair.first)) {
        std::cout << "ERROR Dependencies::GenerateReverseDependencies: "
                     "Internal error reversing deps"
                  << std::endl;
        return ReverseDependencies{};
      }
    }
  }

  if (fauxDependencies.HasCycle()) {
    std::cout << "ERROR Dependencies::GenerateReverseDependencies: Internal "
                 "error: ReverseDependencies has a cycle"
              << std::endl;
    return ReverseDependencies{};
  }

  return ReverseDependencies(fauxDependencies.deps_);
}

bool Dependencies::CycleFromValue(unsigned int value) const {
  std::unordered_set<unsigned int> visited{};
  std::queue<unsigned int> next_queue{};
  next_queue.push(value);

  unsigned int temp;
  while (!next_queue.empty()) {
    temp = next_queue.front();
    next_queue.pop();
    if (visited.find(temp) != visited.end()) {
      return true;
    }

    visited.insert(temp);

    for (unsigned int dependency : GetDependencies(temp)) {
      next_queue.push(dependency);
    }
  }

  return false;
}
