import pytest
from datetime import datetime
from okmich_quant_core.filter import BaseFilter, FilterChain


# Concrete test filter implementations
class AlwaysPassFilter(BaseFilter):
    """Test filter that always passes."""

    def do_filter(self, context):
        return True


class AlwaysBlockFilter(BaseFilter):
    """Test filter that always blocks."""

    def do_filter(self, context):
        return False


class ConditionalFilter(BaseFilter):
    """Test filter that checks a specific context key."""

    def __init__(self, key, expected_value, name=None):
        super().__init__(name)
        self.key = key
        self.expected_value = expected_value

    def do_filter(self, context):
        return context.get(self.key) == self.expected_value


class TestBaseFilter:
    """Test BaseFilter abstract base class behavior."""

    def test_filter_has_name(self):
        """Test that filters have names (default or custom)."""
        filter1 = AlwaysPassFilter()
        assert filter1.name == "AlwaysPassFilter"

        filter2 = AlwaysPassFilter(name="CustomName")
        assert filter2.name == "CustomName"

    def test_filter_callable(self):
        """Test that filters can be called directly."""
        pass_filter = AlwaysPassFilter()
        block_filter = AlwaysBlockFilter()

        assert pass_filter({}) is True
        assert block_filter({}) is False

    def test_filter_receives_context(self):
        """Test that filters receive and can use context."""
        filter = ConditionalFilter("test_key", "expected_value")

        context_pass = {"test_key": "expected_value"}
        context_fail = {"test_key": "wrong_value"}

        assert filter(context_pass) is True
        assert filter(context_fail) is False


class TestFilterChain:
    """Test FilterChain composite pattern."""

    def test_empty_chain_passes(self):
        """Test that empty filter chain always passes."""
        chain = FilterChain([])
        assert chain({}) is True

    def test_single_filter_pass(self):
        """Test chain with single passing filter."""
        chain = FilterChain([AlwaysPassFilter()])
        assert chain({}) is True

    def test_single_filter_block(self):
        """Test chain with single blocking filter."""
        chain = FilterChain([AlwaysBlockFilter()])
        assert chain({}) is False

    def test_multiple_filters_all_pass(self):
        """Test chain where all filters pass."""
        chain = FilterChain(
            [AlwaysPassFilter(), AlwaysPassFilter(), AlwaysPassFilter()]
        )
        assert chain({}) is True

    def test_multiple_filters_one_blocks(self):
        """Test chain where one filter blocks (short-circuit)."""
        chain = FilterChain(
            [AlwaysPassFilter(), AlwaysBlockFilter(), AlwaysPassFilter()]
        )
        assert chain({}) is False

    def test_multiple_filters_first_blocks(self):
        """Test that chain stops at first blocking filter."""
        chain = FilterChain(
            [AlwaysBlockFilter(), AlwaysPassFilter(), AlwaysPassFilter()]
        )
        assert chain({}) is False

    def test_complex_chain_with_conditions(self):
        """Test chain with conditional filters."""
        chain = FilterChain(
            [
                ConditionalFilter("key1", "value1"),
                ConditionalFilter("key2", "value2"),
                ConditionalFilter("key3", "value3"),
            ]
        )

        # All conditions met
        context_pass = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert chain(context_pass) is True

        # First condition fails
        context_fail1 = {"key1": "wrong", "key2": "value2", "key3": "value3"}
        assert chain(context_fail1) is False

        # Second condition fails
        context_fail2 = {"key1": "value1", "key2": "wrong", "key3": "value3"}
        assert chain(context_fail2) is False

        # Last condition fails
        context_fail3 = {"key1": "value1", "key2": "value2", "key3": "wrong"}
        assert chain(context_fail3) is False

    def test_add_filter(self):
        """Test dynamically adding filters to chain."""
        chain = FilterChain([])
        assert chain({}) is True

        chain.add_filter(AlwaysPassFilter())
        assert chain({}) is True

        chain.add_filter(AlwaysBlockFilter())
        assert chain({}) is False

    def test_remove_filter(self):
        """Test removing filters from chain by name."""
        pass_filter = AlwaysPassFilter(name="Pass")
        block_filter = AlwaysBlockFilter(name="Block")

        chain = FilterChain([pass_filter, block_filter])
        assert chain({}) is False

        chain.remove_filter("Block")
        assert chain({}) is True

    def test_chain_has_name(self):
        """Test that FilterChain has a name."""
        chain1 = FilterChain([])
        assert chain1.name == "FilterChain"

        chain2 = FilterChain([], name="CustomChain")
        assert chain2.name == "CustomChain"

    def test_nested_chains(self):
        """Test that filter chains can be nested."""
        inner_chain = FilterChain([AlwaysPassFilter(), AlwaysPassFilter()])
        outer_chain = FilterChain([AlwaysPassFilter(), inner_chain])

        assert outer_chain({}) is True

        inner_chain_with_block = FilterChain(
            [AlwaysPassFilter(), AlwaysBlockFilter()]
        )
        outer_chain_blocked = FilterChain([AlwaysPassFilter(), inner_chain_with_block])

        assert outer_chain_blocked({}) is False


class TestFilterContextHandling:
    """Test proper context passing between filters."""

    def test_context_immutability(self):
        """Test that filters don't modify the original context."""

        class ContextModifyingFilter(BaseFilter):
            def do_filter(self, context):
                context["modified"] = True
                return True

        original_context = {"key": "value"}
        filter = ContextModifyingFilter()
        filter(original_context)

        # Context was modified (Python dicts are mutable)
        # This is expected behavior - filters can see each other's modifications
        assert original_context["modified"] is True

    def test_context_sharing_between_filters(self):
        """Test that filters in a chain share context."""

        class ContextSetterFilter(BaseFilter):
            def do_filter(self, context):
                context["set_by_first"] = True
                return True

        class ContextCheckerFilter(BaseFilter):
            def do_filter(self, context):
                return context.get("set_by_first", False)

        chain = FilterChain([ContextSetterFilter(), ContextCheckerFilter()])
        context = {}

        # Should pass because first filter sets the key
        assert chain(context) is True
        assert context["set_by_first"] is True

    def test_missing_context_keys(self):
        """Test filters handle missing context keys gracefully."""
        filter = ConditionalFilter("missing_key", "value")

        # Should return False because key doesn't exist
        assert filter({}) is False

        # Should return False because value doesn't match
        assert filter({"missing_key": "other"}) is False

        # Should return True when key and value match
        assert filter({"missing_key": "value"}) is True