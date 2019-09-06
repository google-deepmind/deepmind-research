-- Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--    http://www.apache.org/licenses/LICENSE-2.0
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
-- ============================================================================
local make_map = require 'common.make_map'
local custom_observations = require 'decorators.custom_observations'
local debug_observations = require 'decorators.debug_observations'
local game = require 'dmlab.system.game'
local map_maker = require 'dmlab.system.map_maker'
local maze_generation = require 'dmlab.system.maze_generation'
local pickup_decorator = require 'decorators.human_recognisable_pickups'
local random = require 'common.random'
local setting_overrides = require 'decorators.setting_overrides'
local texture_sets = require 'themes.texture_sets'
local themes = require 'themes.themes'
local hrp = require 'common.human_recognisable_pickups'

local DEFAULTS = {
    EPISODE_LENGTH_SECONDS = 15,
    EXPLORE_LENGTH_SECONDS = 5,
    DISTRACTOR_LENGTH_SECONDS = 5,
    REWARD_LENGTH_SECONDS = nil,
    SHOW_KEY_COLOR_SQUARE_SECONDS = 1,
    PROB_APPLE_IN_DISTRACTOR_MAP = 0.3,
    APPLE_REWARD = 5,
    APPLE_REWARD_PROB = 1.0,
    APPLE_EXTRA_REWARD_RANGE = 0,
    GOAL_REWARD = 10,
    DISTRACTOR_ROOM_SIZE = {11, 11},
    DIFFERENT_DISTRACT_ROOM_TEXTURE = false,
    DIFFERENT_REWARD_ROOM_TEXTURE = false,
    KEY_COLOR = {0, 0, 0},
}

local APPLE_ID = 998
local GOAL_ID = 999
local KEY_SPAWN_ID = 1000
local DOOR_ID = 1001

local KEY_CUE_RECTANGLE_WIDTH = 600
local KEY_CUE_RECTANGLE_HEIGHT = 200

-- Table that maps from full decal name to decal index number.
local decalIndices = {}

local EXPLORE_MAP = "exploreMap"
local DISTRACTOR_MAP = "distractorMap"
local REWARD_MAP = "rewardMap"

-- Set texture set for all maps.
local textureSet = texture_sets.PACMAN
local secondTextureSet = texture_sets.TETRIS
local thirdTextureSet = texture_sets.TRON

local REWARD_ROOM =[[
***
*P*
*H*
*G*
***
]]

local OPEN_TWO_ROOM = [[
*********
*********
*PKK*KKK*
*KKKKKKK*
*KKK*KKK*
*********
]]
local N_KEY_POS_IN_TWO_ROOM = 18  -- # of K in OPEN_TWO_ROOM

local function createDistractorMaze(opts)
    -- Example room with height = 2, width = 3
    -- A are possible apple locations (everywhere)
    -- *****
    -- *APA*
    -- *AAA*
    -- *****

    local roomHeight = opts.roomSize[1]
    local roomWidth = opts.roomSize[2]
    centerWidth = 1 + math.ceil(roomWidth / 2)
    local maze = maze_generation:mazeGeneration{
        height = roomHeight + 2,  -- +2 for the two side of walls
        width = roomWidth + 2
    }

    -- Fill the room with 'A' for apples. updateSpawnVars decides where to put.
    for i = 2, roomHeight + 1 do
      for j = 2, roomWidth + 1 do
        maze:setEntityCell(i, j, 'A')
      end
    end
    -- Override one cell with 'P' for spawn point.
    maze:setEntityCell(2, centerWidth, 'P')
    return maze
end

local function numPossibleAppleLocations(distractorRoomSize)
  return distractorRoomSize[1] * distractorRoomSize[2] - 1
end

local factory = {}
game:console('cg_drawScriptRectanglesAlways 1')

function factory.createLevelApi(kwargs)
  kwargs.episodeLengthSeconds = kwargs.episodeLengthSeconds or
                                DEFAULTS.EPISODE_LENGTH_SECONDS
  kwargs.exploreLengthSeconds = kwargs.exploreLengthSeconds or
                                DEFAULTS.EXPLORE_LENGTH_SECONDS
  kwargs.rewardLengthSeconds = kwargs.rewardLengthSeconds or
                               DEFAULTS.REWARD_LENGTH_SECONDS
  kwargs.distractorLengthSeconds = kwargs.distractorLengthSeconds or
                                   DEFAULTS.DISTRACTOR_LENGTH_SECONDS
  kwargs.distractorRoomSize = kwargs.distractorRoomSize or
                              DEFAULTS.DISTRACTOR_ROOM_SIZE

  kwargs.appleReward = kwargs.appleReward or DEFAULTS.APPLE_REWARD
  kwargs.appleRewardProb = kwargs.appleRewardProb or DEFAULTS.APPLE_REWARD_PROB
  kwargs.probAppleInDistractorMap = kwargs.probAppleInDistractorMap or
                                    DEFAULTS.PROB_APPLE_IN_DISTRACTOR_MAP

  kwargs.appleExtraRewardRange =
      kwargs.appleExtraRewardRange or DEFAULTS.APPLE_EXTRA_REWARD_RANGE

  kwargs.differentDistractRoomTexture = kwargs.differentDistractRoomTexture or
                                        DEFAULTS.DIFFERENT_DISTRACT_ROOM_TEXTURE

  kwargs.differentRewardRoomTexture = kwargs.differentRewardRoomTexture or
                                      DEFAULTS.DIFFERENT_REWARD_ROOM_TEXTURE

  kwargs.showKeyColorSquareSeconds = kwargs.showKeyColorSquareSeconds or
                                     DEFAULTS.SHOW_KEY_COLOR_SQUARE_SECONDS
  kwargs.goalReward = kwargs.goalReward or DEFAULTS.GOAL_REWARD
  kwargs.keyColor = kwargs.keyColor or DEFAULTS.KEY_COLOR

  local api = {}

  function api:init(params)
    self:_createExploreMap()
    self:_createDistractorMap()
    self:_createRewardMap()

    local keyInfo = {
        shape='key',
        pattern='solid',
        color1 = kwargs.keyColor,
        color2 = kwargs.keyColor
    }
    self._keyObject = hrp.create(keyInfo)
    self._keyCueRgba = {
        kwargs.keyColor[1]/255,
        kwargs.keyColor[2]/255,
        kwargs.keyColor[3]/255,
        1
    }
  end

  function api:_createRewardMap()
    self._rewardMap = map_maker:mapFromTextLevel{
        mapName = REWARD_MAP,
        entityLayer = REWARD_ROOM,
    }

    -- Create map theme and override default wall decal placement.
    local texture = textureSet
    if kwargs.differentRewardRoomTexture then
      texture = thirdTextureSet
    end
    local rewardMapTheme = themes.fromTextureSet{
        textureSet = texture,
        decalFrequency = 0.0,
        floorModelFrequency = 0.0,
    }

    self._rewardMap = map_maker:mapFromTextLevel{
        mapName = REWARD_MAP,
        entityLayer = REWARD_ROOM,
        theme = rewardMapTheme,
        callback = function (i, j, c, maker)
          local pickup = self:_makePickup(c)
          if pickup then
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:_createExploreMap()
    exploreMapInfo = {map = OPEN_TWO_ROOM}

    -- Create map theme and override default wall decal placement.
    local exploreMapTheme = themes.fromTextureSet{
        textureSet = textureSet,
        decalFrequency = 0.0,
        floorModelFrequency = 0.0,
    }

    self._exploreMap = map_maker:mapFromTextLevel{
        mapName = EXPLORE_MAP,
        entityLayer = exploreMapInfo.map,
        theme = exploreMapTheme,
        callback = function (i, j, c, maker)
          local pickup = self:_makePickup(c)
          if pickup then
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:_createDistractorMap()
    -- Create maze to be converted into map.
    local maze = createDistractorMaze{roomSize = kwargs.distractorRoomSize}

    -- Create map theme with no wall decals.
    local texture = textureSet
    if kwargs.differentDistractRoomTexture then
      texture = secondTextureSet
    end
    local distractorMapTheme = themes.fromTextureSet{
        textureSet = texture,
        decalFrequency = 0.0,
        floorModelFrequency = 0.0,
    }

    self._distractorMap = map_maker:mapFromTextLevel{
        mapName = DISTRACTOR_MAP,
        entityLayer = maze:entityLayer(),
        theme = distractorMapTheme,
        callback = function (i, j, c, maker)
          local pickup = self:_makePickup(c)
          if pickup then
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:start(episode, seed)
    random:seed(seed)

    self._map = nil
    self._time = 0
    self._holdingKey = false
    self._keyPosCount = 0
    self._collectedGoal = false

    if kwargs.distractorLengthSecondsRange then
      self._distractorLen = random:uniformReal(
          kwargs.distractorLengthSecondsRange[1],
          kwargs.distractorLengthSecondsRange[2])
    else
      self._distractorLen = kwargs.distractorLengthSeconds
    end

    -- Sample the key position in phase 1.
    self._keyPosition = random:uniformInt(1, N_KEY_POS_IN_TWO_ROOM)

    -- Default instruction channel to 0 (indicating the rewards in final phase.)
    self.setInstruction(tostring(0))
  end

  function api:filledRectangles(args)
    if self._showKeyCue then
      return {{
          x = 12,
          y = 12,
          width = KEY_CUE_RECTANGLE_WIDTH,
          height = KEY_CUE_RECTANGLE_HEIGHT,
          rgba = self._keyCueRgba
      }}
    end
    return {}
  end

  function api:nextMap()
    -- 1. Decide what is the next map.
    if self._map == nil then
      self._map = EXPLORE_MAP
    elseif self._map == DISTRACTOR_MAP then
      self._map = REWARD_MAP
    elseif self._map == EXPLORE_MAP then
      if self._distractorLen > 0.0 then
        self._map = DISTRACTOR_MAP
      else
        self._map = REWARD_MAP
      end
    elseif self._map == REWARD_MAP then
      -- Stay in distractor map till end of episode.
      self._map = DISTRACTOR_MAP
      self._collectedGoal = true
    end

    -- 2. Set up timeout for the up-coming map.
    if self._map == DISTRACTOR_MAP and self._collectedGoal then
      if not self._timeOut then -- don't override any existing timeout
        self._timeOut = self._time + 0.1
      end
    elseif self._map == EXPLORE_MAP then
      self._timeOut = self._time + kwargs.exploreLengthSeconds
    elseif self._map == DISTRACTOR_MAP then
      self._timeOut = self._time + self._distractorLen
    elseif self._map == REWARD_MAP then
      if kwargs.rewardLengthSeconds then
        self._timeOut = self._time + kwargs.rewardLengthSeconds
      else
        self._timeOut = nil
      end
    end

    return self._map
  end

 -- PICKUP functions ----------------------------------------------------------

  function api:_makePickup(c)
    if c == 'K' then
      return 'key'
    end
    if c == 'G' then
      return 'goal'
    end
    if c == 'A' then
      return 'apple_reward'
    end
  end

  function api:pickup(spawnId)
    if spawnId == GOAL_ID then
      local goalReward = kwargs.goalReward
      game:addScore(goalReward - 10)  -- Offset the default +10 for goal.
      self.setInstruction(tostring(goalReward))
      game:finishMap()
    end
    if spawnId == KEY_SPAWN_ID then
      self._holdingKey = true
      self._holdingKeyTime = self._time  -- When the avatar got the key.
      self._showKeyCue = true
    end

    if spawnId == APPLE_ID then
      if kwargs.appleRewardProb >= 1 or
         random:uniformReal(0, 1) < kwargs.appleRewardProb then
        -- The -1 is to offset the default 1 point for apple in dmlab
        appleReward = kwargs.appleReward +
            random:uniformInt(0, kwargs.appleExtraRewardRange) - 1
        game:addScore(appleReward)
      else
        -- The -1 is to offset the default 1 point for apple in dmlab
        game:addScore(-1)
      end
    end
  end

  -- TRIGGER functions ---------------------------------------------------------

  function api:canTrigger(teleportId, targetName)
    if string.sub(targetName, 1, 4) == 'door' then
      if self._holdingKey then
        return true
      else
        return false
      end
    end
    return true
  end

  function api:trigger(teleportId, targetName)
    if string.sub(targetName, 1, 4) == 'door' then
      -- When door opend, stop showing key cue, and set holding key to false.
      self._showKeyCue = false
      self._holdingKey = false
      return
    end
  end

  function api:hasEpisodeFinished(timeSeconds)
    self._time = timeSeconds

    if self._map == REWARD_MAP or self._collectedGoal then
      return self._timeOut and timeSeconds > self._timeOut
    end

    -- Control the timing of showing key cue.
    if self._holdingKey then
      local showTime = self._time - self._holdingKeyTime
      if showTime > kwargs.showKeyColorSquareSeconds then
        self._showKeyCue = false
      end
    end

    if self._map == EXPLORE_MAP or self._map == DISTRACTOR_MAP then
      if timeSeconds > self._timeOut then
        game:finishMap()
      end
      return false
    end
  end

  -- END TRIGGER functions -----------------------------------------------------

  function api:updateSpawnVars(spawnVars)
    local classname = spawnVars.classname
    if classname == "info_player_start" then
      -- Spawn facing South.
      spawnVars.angle = "-90"
      spawnVars.randomAngleRange = "0"
    elseif classname == "func_door" then
      spawnVars.id = tostring(DOOR_ID)
      spawnVars.wait = "1000000" -- Open the door for long time.
    elseif classname == "goal" then
      spawnVars.id = tostring(GOAL_ID)
    elseif classname == "apple_reward" then
      -- We respawn the avatar to distractor room after reaching goal
      -- there will be no more apples in this case.
      if self._collectedGoal == true then
        return nil
      end
      local useApple = false
      if kwargs.probAppleInDistractorMap > 0 then
        useApple = random:uniformReal(0, 1) < kwargs.probAppleInDistractorMap
      end
      if useApple then
        spawnVars.id = tostring(APPLE_ID)
      else
        return nil
      end
    elseif classname == "key" then
      self._keyPosCount = self._keyPosCount + 1
      if self._keyPosition == self._keyPosCount then
        spawnVars.id = tostring(KEY_SPAWN_ID)
        spawnVars.classname = self._keyObject
      else
        return nil
      end
    end
    return spawnVars
  end

  custom_observations.decorate(api)
  pickup_decorator.decorate(api)
  setting_overrides.decorate{
      api = api,
      apiParams = kwargs,
      decorateWithTimeout = true
  }
  return api
end

return factory
