#include "dependencies.h"

#include <iostream>
#include <queue>

const std::unordered_set<unsigned int> kEmptySet = {};

ReverseDependencies::ReverseDependencies() : reverse_deps_(), deps_() {}

ReverseDependencies::ReverseDependencies(DMap reverse_deps, DMap deps)
    : reverse_deps_(reverse_deps), deps_(deps) {}

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

bool ReverseDependencies::IsEmpty() const {
  return reverse_deps_.empty() || deps_.empty();
}

std::vector<std::vector<unsigned int>>
ReverseDependencies::GetDependenciesOrdered() const {
  std::vector<std::vector<unsigned int>> stages;
  std::vector<unsigned int> stage;
  std::unordered_set<unsigned int> visited;
  std::unordered_set<unsigned int> to_visit;
  std::vector<unsigned int> holding;

  // first use visited as a count
  for (const auto &pair : reverse_deps_) {
    visited.insert(pair.first);
    for (unsigned int value : pair.second) {
      visited.insert(value);
    }
  }

  unsigned int item_count = visited.size();
  visited.clear();

  // set up "to_visit" which will hold candidates for the next stage
  for (const auto &pair : reverse_deps_) {
    to_visit.insert(pair.first);
  }

  // Now get all stages where "visited" accounts for previous stages
  while (visited.size() < item_count) {
    stage.clear();
    // push items in "holding" to "to_visit" that isn't in "visited"
    for (unsigned int rdep : holding) {
      if (visited.find(rdep) == visited.end()) {
        to_visit.insert(rdep);
      }
    }
    holding.clear();

    // check each item in "to_visit" as they are candidates for the stage
    for (auto iter = to_visit.begin(); iter != to_visit.end();) {
      if (visited.find(*iter) != visited.end()) {
        // already in visited, put rdeps in "holding" and continue
        auto rdeps = GetReverseDependents(*iter);
        for (unsigned int reverse_dep : rdeps) {
          holding.push_back(reverse_dep);
        }

        iter = to_visit.erase(iter);
        continue;
      }

      // not in visited, check deps
      std::unordered_set<unsigned int> deps;
      auto deps_iter = deps_.find(*iter);
      if (deps_iter != deps_.end()) {
        deps = deps_iter->second;
      }
      for (unsigned int visited_value : visited) {
        deps.erase(visited_value);
      }
      if (deps.empty()) {
        // no current deps, is a valid candidate, push into "stage"
        stage.push_back(*iter);

        // push future candidates into "holding"
        auto rdeps = GetReverseDependents(*iter);
        for (unsigned int reverse_dep : rdeps) {
          holding.push_back(reverse_dep);
        }

        iter = to_visit.erase(iter);
        continue;
      }

      ++iter;
    }

    // push "stage" into "stages", and also into "visited"
    stages.push_back(stage);
    for (unsigned int dep : stage) {
      visited.insert(dep);
    }
  }

  return stages;
}

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

  return ReverseDependencies(fauxDependencies.deps_, deps_);
}

bool Dependencies::CycleFromValue(unsigned int value) const {
  std::unordered_set<unsigned int> visited{};
  std::queue<unsigned int> next_queue{};
  next_queue.push(value);

  unsigned int temp;
  while (!next_queue.empty()) {
    temp = next_queue.front();
    next_queue.pop();

    for (unsigned int dependency : GetDependencies(temp)) {
      if (dependency == value) {
        return true;
      }
      if (visited.find(dependency) == visited.end()) {
        visited.insert(dependency);
        next_queue.push(dependency);
      }
    }
  }

  return false;
}
