#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Command Sequencer
===================

Manages AI command sequences for the Schwabot trading system.
This module handles the sequencing, execution, and validation of AI-generated
trading commands and strategies.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime

logger = logging.getLogger(__name__)

class CommandStatus(Enum):
    """Status of AI commands."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CommandPriority(Enum):
    """Priority levels for AI commands."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AICommand:
    """Represents an AI-generated command."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: CommandPriority = CommandPriority.MEDIUM
    status: CommandStatus = CommandStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CommandSequence:
    """Represents a sequence of AI commands."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    commands: List[AICommand] = field(default_factory=list)
    status: CommandStatus = CommandStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AICommandSequencer:
    """Manages AI command sequences and execution."""
    
    def __init__(self, max_concurrent_commands: int = 10):
        """Initialize the AI command sequencer."""
        self.max_concurrent_commands = max_concurrent_commands
        self.active_commands: Dict[str, AICommand] = {}
        self.command_sequences: Dict[str, CommandSequence] = {}
        self.command_queue: List[AICommand] = []
        self.execution_history: List[AICommand] = []
        self.running = False
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_commands)
        
        # Command type handlers
        self.command_handlers: Dict[str, Callable[[AICommand], Awaitable[Any]]] = {}
        
        # Statistics
        self.stats = {
            "total_commands": 0,
            "completed_commands": 0,
            "failed_commands": 0,
            "average_execution_time": 0.0
        }
    
    async def start(self):
        """Start the command sequencer."""
        if self.running:
            return
        
        self.running = True
        logger.info("🚀 AI Command Sequencer started")
        
        # Start command processing loop
        asyncio.create_task(self._command_processing_loop())
    
    async def stop(self):
        """Stop the command sequencer."""
        self.running = False
        logger.info("🛑 AI Command Sequencer stopped")
    
    def register_command_handler(self, command_type: str, handler: Callable[[AICommand], Awaitable[Any]]):
        """Register a handler for a specific command type."""
        self.command_handlers[command_type] = handler
        logger.info(f"📝 Registered handler for command type: {command_type}")
    
    async def add_command(self, command: AICommand) -> str:
        """Add a command to the queue."""
        self.command_queue.append(command)
        self.stats["total_commands"] += 1
        
        # Sort queue by priority
        self.command_queue.sort(key=lambda c: c.priority.value, reverse=True)
        
        logger.info(f"➕ Added command {command.id} ({command.command_type}) to queue")
        return command.id
    
    async def add_command_sequence(self, sequence: CommandSequence) -> str:
        """Add a command sequence."""
        self.command_sequences[sequence.id] = sequence
        
        # Add all commands to queue
        for command in sequence.commands:
            await self.add_command(command)
        
        logger.info(f"📋 Added command sequence {sequence.id} with {len(sequence.commands)} commands")
        return sequence.id
    
    async def _command_processing_loop(self):
        """Main command processing loop."""
        while self.running:
            try:
                # Process commands in queue
                if self.command_queue and len(self.active_commands) < self.max_concurrent_commands:
                    command = self.command_queue.pop(0)
                    asyncio.create_task(self._execute_command(command))
                
                # Clean up completed commands
                await self._cleanup_completed_commands()
                
                # Update statistics
                await self._update_statistics()
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"❌ Command processing loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_command(self, command: AICommand):
        """Execute a single command."""
        async with self.execution_semaphore:
            try:
                command.status = CommandStatus.EXECUTING
                command.executed_at = datetime.now()
                self.active_commands[command.id] = command
                
                logger.info(f"⚡ Executing command {command.id} ({command.command_type})")
                
                # Execute command
                start_time = time.time()
                result = await self._execute_command_handler(command)
                execution_time = time.time() - start_time
                
                # Update command
                command.status = CommandStatus.COMPLETED
                command.completed_at = datetime.now()
                command.result = result
                command.metadata["execution_time"] = execution_time
                
                self.stats["completed_commands"] += 1
                logger.info(f"✅ Command {command.id} completed in {execution_time:.2f}s")
                
            except Exception as e:
                command.status = CommandStatus.FAILED
                command.completed_at = datetime.now()
                command.error = str(e)
                
                self.stats["failed_commands"] += 1
                logger.error(f"❌ Command {command.id} failed: {e}")
            
            finally:
                # Remove from active commands
                if command.id in self.active_commands:
                    del self.active_commands[command.id]
                
                # Add to execution history
                self.execution_history.append(command)
    
    async def _execute_command_handler(self, command: AICommand) -> Any:
        """Execute command using registered handler."""
        handler = self.command_handlers.get(command.command_type)
        
        if handler is None:
            raise ValueError(f"No handler registered for command type: {command.command_type}")
        
        return await handler(command)
    
    async def _cleanup_completed_commands(self):
        """Clean up completed commands from history."""
        # Keep only last 1000 commands in history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    async def _update_statistics(self):
        """Update execution statistics."""
        if self.stats["completed_commands"] > 0:
            total_time = sum(
                cmd.metadata.get("execution_time", 0) 
                for cmd in self.execution_history 
                if cmd.status == CommandStatus.COMPLETED
            )
            self.stats["average_execution_time"] = total_time / self.stats["completed_commands"]
    
    def get_command_status(self, command_id: str) -> Optional[AICommand]:
        """Get the status of a specific command."""
        # Check active commands
        if command_id in self.active_commands:
            return self.active_commands[command_id]
        
        # Check execution history
        for command in self.execution_history:
            if command.id == command_id:
                return command
        
        return None
    
    def get_sequence_status(self, sequence_id: str) -> Optional[CommandSequence]:
        """Get the status of a command sequence."""
        return self.command_sequences.get(sequence_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self.stats,
            "active_commands": len(self.active_commands),
            "queued_commands": len(self.command_queue),
            "total_sequences": len(self.command_sequences)
        }

# Global instance
_sequencer = AICommandSequencer()

async def sequence_ai_command(command: AICommand) -> str:
    """Add an AI command to the sequencer."""
    return await _sequencer.add_command(command)

async def update_command_sequence_result(sequence_id: str, command_id: str, result: Any):
    """Update the result of a command in a sequence."""
    sequence = _sequencer.get_sequence_status(sequence_id)
    if sequence:
        for command in sequence.commands:
            if command.id == command_id:
                command.result = result
                command.status = CommandStatus.COMPLETED
                command.completed_at = datetime.now()
                break

# Convenience functions for common command types
async def create_trading_command(symbol: str, action: str, amount: float, **kwargs) -> AICommand:
    """Create a trading command."""
    return AICommand(
        command_type="trading",
        parameters={
            "symbol": symbol,
            "action": action,
            "amount": amount,
            **kwargs
        },
        priority=CommandPriority.HIGH
    )

async def create_analysis_command(symbol: str, analysis_type: str, **kwargs) -> AICommand:
    """Create an analysis command."""
    return AICommand(
        command_type="analysis",
        parameters={
            "symbol": symbol,
            "analysis_type": analysis_type,
            **kwargs
        },
        priority=CommandPriority.MEDIUM
    )

async def create_risk_command(action: str, parameters: Dict[str, Any]) -> AICommand:
    """Create a risk management command."""
    return AICommand(
        command_type="risk_management",
        parameters={
            "action": action,
            **parameters
        },
        priority=CommandPriority.CRITICAL
    ) 