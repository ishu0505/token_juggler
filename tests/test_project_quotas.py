"""Project quota modes on a shared deployment: cap, reserve, reserve + lend.

Deployment m@a allows rpm 4. Every test runs on both backends."""

from tests.conftest import make_registry
from tokenjuggler.limiter import Cost, Limiter

ONE = Cost(1, 1)


def setup(backend, projects):
    registry = make_registry(limits={"rpm": 4}, projects=projects,
                             deployments=[{"account": "a", "model_id": "x"}])
    dep = registry.model("m").deployments[0]

    async def grants(project: str, tries: int = 10) -> int:
        limiter = Limiter(registry, backend, project=project)
        got = 0
        for _ in range(tries):
            hold, _ = await limiter.acquire([dep], [ONE])
            got += hold is not None
        return got

    return grants, registry, dep


async def test_a_hard_reserve_is_kept_for_its_owner(backend):
    grants, *_ = setup(backend, {"batch": {"reserve": 0.5}})
    assert await grants("web") == 2      # only the unreserved pool (50%)
    assert await grants("batch") == 2    # its reserve is still intact


async def test_a_reserve_holder_uses_its_reserve_then_the_pool(backend):
    grants, *_ = setup(backend, {"batch": {"reserve": 0.25}})
    assert await grants("batch") == 4    # 1 from its reserve + 3 from the pool
    assert await grants("web") == 0


async def test_lend_idle_lets_others_borrow_an_unused_reserve(backend):
    grants, *_ = setup(backend, {"batch": {"reserve": 0.5, "lend_idle": True}})
    assert await grants("web") == 4      # 2 from the pool + 2 borrowed
    assert await grants("batch") == 0    # borrowed away; refills over the minute


async def test_reserve_without_any_pool_leaves_unlisted_projects_nothing(backend):
    grants, *_ = setup(backend, {"a1": {"reserve": 0.5}, "a2": {"reserve": 0.5}})
    assert await grants("stranger") == 0
    assert await grants("a1") == 2 and await grants("a2") == 2


async def test_cap_and_reserve_combine(backend):
    grants, *_ = setup(backend, {"batch": {"reserve": 0.25, "cap": 0.5}})
    assert await grants("batch") == 2    # reserve 1 + pool 1, then the cap stops it
    assert await grants("web") == 2      # the pool still has 2 left


async def test_settle_refunds_the_source_that_paid(backend):
    registry = make_registry(limits={"output_tpm": 1000}, projects={"batch": {"reserve": 0.5}},
                             deployments=[{"account": "a", "model_id": "x"}])
    dep = registry.model("m").deployments[0]
    web = Limiter(registry, backend, project="web")
    big = Cost(1, 500)
    hold, _ = await web.acquire([dep], [big])            # fills the 500-token pool
    assert (await web.acquire([dep], [big]))[0] is None
    await web.settle(hold, actual=Cost(1, 10))            # 490 back to the POOL
    assert (await web.acquire([dep], [Cost(1, 400)]))[0] is not None
    # batch's reserve was never touched.
    batch = Limiter(registry, backend, project="batch")
    assert (await batch.acquire([dep], [big]))[0] is not None


async def test_no_reservations_means_no_change_from_the_simple_setup(backend):
    grants, *_ = setup(backend, {"web": {"cap": 0.5}})
    assert await grants("web") == 2
    assert await grants("other") == 2    # caps don't set anything aside
