#pragma once

#include <rocksdb/sst_partitioner.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/compaction_filter.h>

#include <string>
#include <cstring>
#include <memory>

namespace vq {

class VectorSstPartitioner : public rocksdb::SstPartitioner {
public:
    explicit VectorSstPartitioner(int prefix_len)
        : prefix_len_(prefix_len) {}

    const char* Name() const override { return "VectorSstPartitioner"; }

    PartitionerResult ShouldPartition(const SstPartitioner::Request& request) override {
        if (request.current_user_key.size() >= (size_t)prefix_len_ &&
            request.next_user_key.size() >= (size_t)prefix_len_) {
            auto cur_prefix = request.current_user_key.substr(0, prefix_len_);
            auto next_prefix = request.next_user_key.substr(0, prefix_len_);
            if (cur_prefix != next_prefix) {
                return PartitionerResult::kPartition;
            }
        }
        return PartitionerResult::kNotPartition;
    }

    bool CanDoTrivialMove(const rocksdb::Slice& smallest_user_key,
                          const rocksdb::Slice& largest_user_key) const override {
        return false;
    }

private:
    int prefix_len_;
};

class VectorSstPartitionerFactory : public rocksdb::SstPartitionerFactory {
public:
    explicit VectorSstPartitionerFactory(int prefix_len)
        : prefix_len_(prefix_len) {}

    const char* Name() const override { return "VectorSstPartitionerFactory"; }

    std::unique_ptr<rocksdb::SstPartitioner> CreatePartitioner(
        const rocksdb::SstPartitioner::Context& context) const override {
        return std::make_unique<VectorSstPartitioner>(prefix_len_);
    }

private:
    int prefix_len_;
};

class VectorMergeOperator : public rocksdb::MergeOperator {
public:
    const char* Name() const override { return "VectorMergeOperator"; }

    bool FullMergeV2(const MergeOperationInput& merge_in,
                     MergeOperationOutput* merge_out) const override {
        uint32_t count = 0;
        uint32_t start_id = 0;
        if (merge_in.existing_value && merge_in.existing_value->size() >= 8) {
            memcpy(&count, merge_in.existing_value->data(), 4);
            memcpy(&start_id, merge_in.existing_value->data() + 4, 4);
        }
        for (auto& operand : merge_in.operand_list) {
            if (operand.size() >= 4) {
                uint32_t delta;
                memcpy(&delta, operand.data(), 4);
                count += delta;
            }
        }
        std::string result(8, '\0');
        memcpy(&result[0], &count, 4);
        memcpy(&result[4], &start_id, 4);
        merge_out->new_value = result;
        return true;
    }

    bool PartialMergeMulti(const rocksdb::Slice& key,
                           const std::deque<rocksdb::Slice>& operand_list,
                           std::string* new_value,
                           rocksdb::Logger* logger) const override {
        uint32_t total_delta = 0;
        for (auto& op : operand_list) {
            if (op.size() >= 4) {
                uint32_t delta;
                memcpy(&delta, op.data(), 4);
                total_delta += delta;
            }
        }
        new_value->resize(4);
        memcpy(&(*new_value)[0], &total_delta, 4);
        return true;
    }
};

class VectorCompactionFilter : public rocksdb::CompactionFilter {
public:
    const char* Name() const override { return "VectorCompactionFilter"; }

    Decision FilterV2(int level, const rocksdb::Slice& key, ValueType value_type,
                      const rocksdb::Slice& existing_value, std::string* new_value,
                      std::string* skip_until) const override {
        return Decision::kKeep;
    }
};

class VectorCompactionFilterFactory : public rocksdb::CompactionFilterFactory {
public:
    const char* Name() const override { return "VectorCompactionFilterFactory"; }

    std::unique_ptr<rocksdb::CompactionFilter> CreateCompactionFilter(
        const rocksdb::CompactionFilter::Context& context) override {
        return std::make_unique<VectorCompactionFilter>();
    }
};

} // namespace vq
