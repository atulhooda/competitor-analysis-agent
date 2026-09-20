from app.crawling.ratelimit import HostRateLimiter
from tests.fakesite import FakeClock


async def test_same_host_waits_between_requests(clock: FakeClock) -> None:
    limiter = HostRateLimiter(clock=clock, sleep=clock.sleep)
    for _ in range(3):
        async with limiter.slot("acme.test", 2.0):
            pass
    assert clock.sleeps == [2.0, 2.0]


async def test_elapsed_time_counts_toward_the_delay(clock: FakeClock) -> None:
    limiter = HostRateLimiter(clock=clock, sleep=clock.sleep)
    async with limiter.slot("acme.test", 3.0):
        pass
    clock.now += 2.0
    async with limiter.slot("acme.test", 3.0):
        pass
    assert clock.sleeps == [1.0]


async def test_different_hosts_do_not_wait_for_each_other(clock: FakeClock) -> None:
    limiter = HostRateLimiter(clock=clock, sleep=clock.sleep)
    async with limiter.slot("a.test", 5.0):
        pass
    async with limiter.slot("b.test", 5.0):
        pass
    assert clock.sleeps == []
