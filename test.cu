#include <atomic>
#include <cstdlib>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

#include "managed_ptr.hpp"
#include "gtest/gtest.hpp"

std::atomic<unsigned> A_counter(0);

struct A {
    int x = 100;
    int y = 200;

    A() = default;

    A(int X, int Y): x(X), y(Y)
    {}
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(managed_ptr, default_constructor) {
    managed_ptr<int> p;

    // default construction implies that no memory is allocated, hence the
    // managed_ptr should evaluate to false
    EXPECT_FALSE(p);
}

TEST(managed_ptr, make_managed_ptr) {
    // check that the default constructor is called
    {
        auto a = make_managed_ptr<A>();
        EXPECT_TRUE(a);
        EXPECT_EQ(100, a->x);
        EXPECT_EQ(200, a->y);

        // int should be default constructed to zero, i.e. int()==0
        auto i = make_managed_ptr<int>();
        EXPECT_TRUE(i);
        EXPECT_EQ(0, *i);
    }
}

TEST(managed_ptr, swap) {
    auto a1 = make_managed_ptr<A>(1, 2);
    auto a2 = make_managed_ptr<A>(-1, -2);
    EXPECT_TRUE(a1);
    EXPECT_TRUE(a2);

    auto p1 = a1.get();
    auto p2 = a2.get();

    EXPECT_EQ(1, a1->x);
    EXPECT_EQ(2, a1->y);
    EXPECT_EQ(-1, a2->x);
    EXPECT_EQ(-2, a2->y);

    std::swap(a1, a2);
    EXPECT_EQ(-1, a1->x);
    EXPECT_EQ(-2, a1->y);
    EXPECT_EQ(1, a2->x);
    EXPECT_EQ(2, a2->y);

    EXPECT_EQ(p1, a2.get());
    EXPECT_EQ(p2, a1.get());
}

TEST(managed_ptr, move_constructor) {
    auto a1 = make_managed_ptr<A>(1, 2);
    auto p1 = a1.get();
    managed_ptr<A> a2 = std::move(a1);

    EXPECT_FALSE(a1);
    EXPECT_TRUE(a2);

    EXPECT_EQ(p1, a2.get());

    EXPECT_EQ(1, a2->x);
    EXPECT_EQ(2, a2->y);
}

__global__
void kernel_ref(A& a) {
    a.x = 23;
    a.y = 32;
}

TEST(managed_ptr, as_kernel_argument) {
    auto a = make_managed_ptr<A>(1, 2);
    EXPECT_TRUE(a);
    EXPECT_EQ(1, a->x);
    EXPECT_EQ(2, a->y);

    kernel_ref<<<1, 1>>>(*a);
    a.synchronize();

    // after synchronization, check that updates made in kernel are visible to the host
    EXPECT_EQ(23, a->x);
    EXPECT_EQ(32, a->y);
}

// A type with a managed_ptr member
// For testing the use of managed_ptr inside kernels, which is required for "deep copy"
struct nested {
    int x;
    managed_ptr<int> inner;

    nested(int X, int N):
        x{X}, inner{make_managed_ptr<int>(N)}
    {}
};

__global__
void kernel_nested(nested& n) {
    n.x = 100;
    *n.inner = 200;
}

// Tests that kernels can access and modify memory managed by managed_ptr
// via managed_ptr member functions
TEST(managed_ptr, nested) {
    auto ptr = make_managed_ptr<nested>(1, 2);

    EXPECT_TRUE(ptr);
    EXPECT_TRUE(ptr->inner);

    kernel_nested<<<1, 1>>>(*ptr);
    ptr.synchronize();

    EXPECT_EQ(100, ptr->x);
    EXPECT_EQ(200, *ptr->inner);
}

// test that std::move and std::swap can be used on types that have manged_ptr members
TEST(managed_ptr, deep_copy) {
    {
        auto p1 = make_managed_ptr<nested>(1, 2);
        EXPECT_TRUE(p1);
        EXPECT_TRUE(p1->inner);

        managed_ptr<nested> p2 = std::move(p1);

        EXPECT_FALSE(p1);
        EXPECT_TRUE(p2);
        EXPECT_TRUE(p2->inner);
        EXPECT_EQ(1, p2->x);
        EXPECT_EQ(2, *p2->inner);
    }

    {
        auto p1 = make_managed_ptr<nested>(1, 2);
        EXPECT_TRUE(p1);
        EXPECT_TRUE(p1->inner);
        EXPECT_EQ(1, p1->x);
        EXPECT_EQ(2, *p1->inner);

        auto p2 = make_managed_ptr<nested>(2, 1);
        EXPECT_TRUE(p2);
        EXPECT_TRUE(p2->inner);
        EXPECT_EQ(2, p2->x);
        EXPECT_EQ(1, *p2->inner);

        std::swap(p1, p2);

        EXPECT_EQ(2, p1->x);
        EXPECT_EQ(1, *p1->inner);
        EXPECT_EQ(1, p2->x);
        EXPECT_EQ(2, *p2->inner);
    }
}
