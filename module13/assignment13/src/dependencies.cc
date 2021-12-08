#include "dependencies.h"

#include <queue>

const std::unordered_set<unsigned int> Dependencies::empty_set_ = {};

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

const std::unordered_set<unsigned int> &Dependencies::GetDependencies(
    unsigned int from) const {
  auto iter = deps_.find(from);
  if (iter == deps_.end()) {
    return empty_set_;
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
